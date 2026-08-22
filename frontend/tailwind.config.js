/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        app: {
          bg: '#080B14',
          sidebar: '#0B1020',
          card: '#111827',
          border: '#1F2937',
          purple: '#8B5CF6',
          indigo: '#6366F1',
          muted: '#94A3B8'
        }
      }
    },
  },
  plugins: [],
}
