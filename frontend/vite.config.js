import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/languages': 'http://127.0.0.1:8000',
      '/sample-files': 'http://127.0.0.1:8000',
      '/samples': 'http://127.0.0.1:8000',
      '/synthesize': 'http://127.0.0.1:8000',
      '/translate-audio': 'http://127.0.0.1:8000',
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
})
