import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0", port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("/react/") || id.includes("/react-dom/") || id.includes("scheduler/")) {
            return "react-vendor";
          }
          if (id.includes("/antd/") || id.includes("/@ant-design/")) {
            return "antd-vendor";
          }
          if (id.includes("/echarts/") || id.includes("/echarts-for-react/")) {
            return "chart-vendor";
          }
          return undefined;
        },
      },
    },
  },
});
