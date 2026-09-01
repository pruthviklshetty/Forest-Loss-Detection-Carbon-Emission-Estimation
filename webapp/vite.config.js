import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API base is read from VITE_API_URL at build time (see .env.example);
// default is the local FastAPI dev server. `base: './'` keeps built asset
// paths relative so dist/ can be served from any sub-path or static host.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: { port: 5173 },
  build: { outDir: 'dist', assetsDir: 'assets' },
})
