import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Same-origin API: the browser only talks to Vite, so no CORS config is needed.
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
