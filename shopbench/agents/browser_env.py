"""Playwright browser environment for the store, with pluggable observation strategies.

Observation modes (the context-engineering variable):
  html           full page HTML (scripts/styles removed), interactive elements tagged data-ref="eN"
  axtree         accessibility-style tree of the whole page: roles, names, values, all text
  axtree_pruned  the same tree, pruned: boilerplate collapsed, long text truncated,
                 long option lists summarized, empty containers dropped

Actions reference elements by the refs shown in the observation (e.g. click ref="e12").
Refs are re-assigned on every observation in DOM order.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from playwright.sync_api import Browser, Page, sync_playwright

WALK_JS = r"""
() => {
  let n = 0;
  document.querySelectorAll('[data-ref]').forEach(e => e.removeAttribute('data-ref'));
  const SKIP = new Set(['SCRIPT','STYLE','NOSCRIPT','TEMPLATE','HEAD','META','LINK']);
  const visible = el => {
    const s = getComputedStyle(el);
    if (s.display === 'none' || s.visibility === 'hidden') return false;
    const r = el.getBoundingClientRect();
    // the store hides some labels off-screen for screen readers; keep them in the tree
    return r.width > 0 || r.height > 0 || s.position === 'absolute';
  };
  const labelFor = el => {
    if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
    if (el.id) { const l = document.querySelector(`label[for="${el.id}"]`); if (l) return l.innerText.trim(); }
    const pl = el.closest('label'); if (pl) return pl.innerText.trim();
    return el.getAttribute('placeholder') || el.getAttribute('name') || '';
  };
  const roleOf = el => {
    const r = el.getAttribute('role'); if (r) return r;
    const t = el.tagName;
    if (t === 'A' && el.hasAttribute('href')) return 'link';
    if (t === 'BUTTON') return 'button';
    if (t === 'INPUT') {
      const ty = (el.getAttribute('type') || 'text').toLowerCase();
      if (ty === 'hidden') return null;
      if (ty === 'submit' || ty === 'button') return 'button';
      if (ty === 'checkbox') return 'checkbox';
      if (ty === 'radio') return 'radio';
      if (ty === 'search') return 'searchbox';
      if (ty === 'number') return 'spinbutton';
      return 'textbox';
    }
    if (t === 'SELECT') return 'combobox';
    if (t === 'TEXTAREA') return 'textbox';
    if (/^H[1-6]$/.test(t)) return 'heading';
    return ({UL:'list', OL:'list', LI:'listitem', TABLE:'table', TR:'row', TD:'cell', TH:'columnheader',
             NAV:'navigation', MAIN:'main', HEADER:'banner', FOOTER:'contentinfo', ASIDE:'complementary', FORM:'form',
             SECTION:'region', P:'paragraph', LABEL:'label', THEAD:'rowgroup', TBODY:'rowgroup'})[t] || 'generic';
  };
  const INTERACTIVE = new Set(['link','button','checkbox','radio','searchbox','spinbutton','textbox','combobox']);
  const walk = el => {
    if (SKIP.has(el.tagName.toUpperCase()) || !visible(el)) return null;
    if (el.getAttribute('aria-hidden') === 'true') return null;   // hidden from assistive tech
    if (el.tagName === 'IMG') return el.alt ? {role: 'img', name: el.alt, children: []} : null;
    const role = roleOf(el);
    if (role === null) return null;
    const node = {role, children: []};
    if (INTERACTIVE.has(role)) {
      const ref = 'e' + (++n);
      el.setAttribute('data-ref', ref);
      node.ref = ref;
      if (role === 'link' || role === 'button') {
        node.name = (el.getAttribute('aria-label') || el.innerText || el.value || '').trim().replace(/\s+/g, ' ');
        if (role === 'link') node.href = el.getAttribute('href');
      } else {
        node.name = labelFor(el);
        if (role === 'combobox') {
          node.value = el.options[el.selectedIndex] ? el.options[el.selectedIndex].text : '';
          node.options = Array.from(el.options).map(o => o.text);
        } else if (role === 'checkbox' || role === 'radio') {
          node.checked = el.checked;
        } else {
          node.value = el.value;
        }
      }
      return node;
    }
    if (role === 'heading') {
      node.level = +el.tagName[1];
      node.name = el.innerText.trim().replace(/\s+/g, ' ');
      return node;
    }
    // a <label for=...> is already the accessible name of its control
    if (el.tagName === 'LABEL' && el.htmlFor && document.getElementById(el.htmlFor)) return null;
    if (el.getAttribute('aria-label') && ['form','navigation','region'].includes(role)) node.name = el.getAttribute('aria-label');
    if (el.tagName === 'SECTION' && el.getAttribute('aria-label')) node.role = 'region';
    for (const c of el.childNodes) {
      if (c.nodeType === 3) {
        const t = c.textContent.replace(/\s+/g, ' ').trim();
        if (t) node.children.push({role: 'text', name: t});
      } else if (c.nodeType === 1) {
        const k = walk(c); if (k) node.children.push(k);
      }
    }
    if (role === 'label') return {role: 'generic', children: node.children};
    return node;
  };
  return walk(document.body);
}
"""

CLEAN_HTML_JS = r"""
() => {
  const clone = document.documentElement.cloneNode(true);
  clone.querySelectorAll('script,style,noscript,meta,link').forEach(e => e.remove());
  return clone.outerHTML;
}
"""


def _merge_text(children: list[dict]) -> list[dict]:
    out: list[dict] = []
    for c in children:
        if c["role"] == "text" and out and out[-1]["role"] == "text":
            out[-1] = {"role": "text", "name": out[-1]["name"] + " " + c["name"]}
        else:
            out.append(c)
    return out


def _iter(n: dict):
    yield n
    for c in n.get("children", []):
        yield from _iter(c)


def render_tree(node: dict, pruned: bool) -> str:
    lines: list[str] = []
    TRUNC = 240

    def fmt(n: dict) -> str:
        role = n["role"]
        name = n.get("name", "")
        if pruned and len(name) > TRUNC:
            name = name[:TRUNC] + "…"
        s = f'{role} "{name}"' if name else role
        if role == "heading":
            s += f" [level={n['level']}]"
        if "ref" in n:
            s += f" [ref={n['ref']}]"
        if role == "link" and not pruned and n.get("href"):
            s += f" [url={n['href']}]"
        if "value" in n and n["value"] not in (None, ""):
            s += f' value="{n["value"]}"'
        if "checked" in n:
            s += " [checked]" if n["checked"] else " [unchecked]"
        if "options" in n:
            opts = n["options"]
            if pruned and len(opts) > 8:
                s += f" options=[{', '.join(opts[:6])}, … {len(opts) - 6} more]"
            else:
                s += f" options=[{', '.join(opts)}]"
        return s

    def rec(n: dict, depth: int) -> None:
        kids = _merge_text(n.get("children", []))
        role = n["role"]
        if pruned:
            if role == "img":
                return  # alt text duplicates the adjacent link
            # site-wide navigation and footers: one summary line instead of every link
            if role in ("navigation", "contentinfo"):
                links = [x for x in _iter(n) if x["role"] == "link"]
                if len(links) > 12:
                    label = n.get("name") or role
                    lines.append("  " * depth + f'- {role} "{label}" ({len(links)} links omitted; first: '
                                 + ", ".join(f'"{x["name"]}" [ref={x["ref"]}]' for x in links[:4]) + ")")
                    return
            # a list item holding one link plus text becomes a single line
            if role == "listitem":
                flat = [x for x in _iter(n) if x is not n and x["role"] in ("link", "text", "button")]
                link = [x for x in flat if x["role"] == "link"]
                if len(link) == 1:
                    txt = " | ".join(x["name"] for x in flat if x["role"] == "text")
                    nm = link[0]["name"]
                    nm = nm if len(nm) <= 150 else nm[:150] + "…"
                    lines.append("  " * depth + f'- listitem: link "{nm}" [ref={link[0]["ref"]}] | {txt[:TRUNC]}')
                    return
            # collapse structural wrappers that carry no name; drop empty containers
            if role in ("generic", "rowgroup", "paragraph", "region", "main", "cell", "label") and "ref" not in n and not n.get("name"):
                for k in kids:
                    rec(k, depth)
                return
            if role in ("list", "table", "row") and not kids:
                return
            # rows and list items with only text become one line
            if role in ("listitem", "row") and all(k["role"] == "text" for k in kids) and kids:
                lines.append("  " * depth + f'{role} "' + " ".join(k["name"] for k in kids)[:TRUNC * 2] + '"')
                return
        if role == "text":
            t = n["name"]
            if pruned and len(t) > TRUNC:
                t = t[:TRUNC] + "…"
            lines.append("  " * depth + f'text "{t}"')
            return
        lines.append("  " * depth + "- " + fmt(n))
        for k in kids:
            rec(k, depth + 1)

    rec(node, 0)
    return "\n".join(lines)


@dataclass
class ActionResult:
    ok: bool
    message: str


class BrowserEnv:
    def __init__(self, base_url: str, obs_mode: str = "axtree_pruned", max_obs_chars: int = 60000,
                 headless: bool = True):
        assert obs_mode in ("html", "axtree", "axtree_pruned"), obs_mode
        self.base_url = base_url.rstrip("/")
        self.obs_mode = obs_mode
        self.max_obs_chars = max_obs_chars
        self._pw = sync_playwright().start()
        self.browser: Browser = self._pw.chromium.launch(headless=headless)
        self.page: Page | None = None
        self._ctx = None

    def reset(self, episode_id: str) -> str:
        if self._ctx:
            self._ctx.close()
        self._ctx = self.browser.new_context(viewport={"width": 1280, "height": 900})
        self.page = self._ctx.new_page()
        self.page.set_default_timeout(8000)
        self.page.goto(f"{self.base_url}/start?episode={episode_id}")
        return self.observe()

    def close(self) -> None:
        try:
            if self._ctx:
                self._ctx.close()
            self.browser.close()
        finally:
            self._pw.stop()

    # ------------------------------------------------------------------ observation
    def observe(self) -> str:
        assert self.page is not None
        tree = self.page.evaluate(WALK_JS)  # also (re)assigns data-ref attributes
        url = urlparse(self.page.url)
        header = f"URL: {url.path}{('?' + url.query) if url.query else ''}\nTitle: {self.page.title()}\n"
        if self.obs_mode == "html":
            body = re.sub(r"\s+", " ", self.page.evaluate(CLEAN_HTML_JS))
        else:
            body = render_tree(tree, pruned=self.obs_mode == "axtree_pruned")
        if len(body) > self.max_obs_chars:
            body = body[: self.max_obs_chars] + f"\n[... observation truncated at {self.max_obs_chars} characters]"
        return header + body

    # ------------------------------------------------------------------ actions
    def _loc(self, ref: str):
        assert self.page is not None
        loc = self.page.locator(f'[data-ref="{ref}"]')
        if loc.count() != 1:
            raise ValueError(f"No element with ref {ref} on the current page. Use a ref from the latest observation.")
        return loc

    def act(self, name: str, args: dict) -> ActionResult:
        assert self.page is not None
        try:
            if name == "click":
                self._loc(args["ref"]).click()
                self.page.wait_for_load_state("load")
                return ActionResult(True, "clicked")
            if name == "type":
                loc = self._loc(args["ref"])
                loc.fill(str(args.get("text", "")))
                if args.get("submit"):
                    loc.press("Enter")
                    self.page.wait_for_load_state("load")
                return ActionResult(True, "typed")
            if name == "select":
                self._loc(args["ref"]).select_option(label=str(args["option"]))
                return ActionResult(True, "selected")
            if name == "goto":
                target = urljoin(self.base_url + "/", str(args["url"]).lstrip("/"))
                if not target.startswith(self.base_url):
                    return ActionResult(False, "Navigation outside the store is not allowed.")
                if urlparse(target).path.startswith(("/admin", "/api", "/start")):
                    return ActionResult(False, "That path is not available to shoppers.")
                self.page.goto(target)
                return ActionResult(True, "navigated")
            if name == "back":
                self.page.go_back()
                return ActionResult(True, "went back")
            return ActionResult(False, f"Unknown browser action {name}")
        except Exception as e:  # surface errors to the agent instead of crashing the episode
            msg = str(e).split("\n")[0]
            return ActionResult(False, f"Action failed: {msg[:300]}")


BROWSER_TOOLS = [
    {"name": "click", "description": "Click a link, button, or checkbox identified by its ref from the latest observation.",
     "input_schema": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}},
    {"name": "type", "description": "Replace the text in an input box. Set submit=true to press Enter afterwards.",
     "input_schema": {"type": "object", "properties": {"ref": {"type": "string"}, "text": {"type": "string"},
                                                       "submit": {"type": "boolean"}}, "required": ["ref", "text"]}},
    {"name": "select", "description": "Choose an option (by its visible label) in a dropdown.",
     "input_schema": {"type": "object", "properties": {"ref": {"type": "string"}, "option": {"type": "string"}},
                      "required": ["ref", "option"]}},
    {"name": "goto", "description": "Navigate to a path on the store, e.g. /cart or /search?q=usb+c+cable&sort=price_asc.",
     "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"name": "back", "description": "Go back to the previous page.",
     "input_schema": {"type": "object", "properties": {}}},
]


def _debug(base_url: str, episode: str, mode: str) -> None:  # pragma: no cover
    env = BrowserEnv(base_url, mode)
    print(env.reset(episode))
    env.close()


if __name__ == "__main__":  # pragma: no cover
    import sys
    _debug(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "axtree_pruned")
