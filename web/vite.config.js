import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev (npm run dev) the UI runs on :5173 and proxies API/audio calls to
// the Python server on :8000. In production the Python server serves the
// built files from web/dist directly.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/audio": "http://127.0.0.1:8000",
    },
  },
});
