import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { unified } from '@astrojs/markdown-remark';
// Make code examples keyboard-scrollable in static HTML, including without JS.
function accessibleCodeBlocks() {
  return (tree) => {
    function visit(node) {
      if (node.type === 'element' && node.tagName === 'pre') {
        node.properties = { ...node.properties, tabIndex: 0, 'aria-label': 'Code example' };
      }
      node.children?.forEach(visit);
    }
    visit(tree);
  };
}
export default defineConfig({ output: 'static', site: process.env.SITE_URL || 'https://giacolees.github.io', markdown: { processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex, accessibleCodeBlocks] }) } });
