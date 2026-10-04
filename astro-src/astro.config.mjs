// @ts-check
import { defineConfig } from 'astro/config';

export default defineConfig({
  output: 'static',
  redirects: {
    '/apokalyi': '/stillness',
    '/apokalyi/[slug]': '/stillness/[slug]',
  },
  markdown: {
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
      defaultColor: false,
    },
  },
});
