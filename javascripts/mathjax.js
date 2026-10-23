// MathJax configuration for the docs.
//
// Markdown pages go through pymdownx.arithmatex, which emits \( ... \) and
// \[ ... \]; notebook cells rendered by mkdocs-jupyter keep raw $ ... $ and
// $$ ... $$.  Register both delimiter styles.  Code and <pre> blocks are
// skipped by MathJax by default, so dollar signs in code are left alone.
window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    // mkdocs-jupyter stores a plain-text copy of every code cell in a hidden
    // div for its copy button; keep MathJax out of it.
    ignoreHtmlClass: "clipboard-copy-txt|tex2jax_ignore",
  },
};
