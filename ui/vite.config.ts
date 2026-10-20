import { defineConfig } from 'vitest/config';

// Static SPA for S3 + CloudFront. `base: './'` keeps asset URLs relative so the bundle works
// at the distribution root or under a path prefix. No runtime CDN assets.
export default defineConfig({
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
    sourcemap: false,
    assetsInlineLimit: 0,
  },
  test: {
    environment: 'jsdom',
    include: ['tests/**/*.test.ts'],
    restoreMocks: true,
  },
});
