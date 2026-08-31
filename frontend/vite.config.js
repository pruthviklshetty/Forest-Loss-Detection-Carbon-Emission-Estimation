import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: './' -> relative asset paths, so the built dist/ works when served from
// a sub-path (GitHub Pages project site) or embedded. fs.allow -> let the build
// import the committed JSON from ../results (outside this project root).
export default defineConfig({
  plugins: [react()],
  base: './',
  server: { fs: { allow: ['..'] } },
  build: { outDir: 'dist', assetsDir: 'assets' },
})
