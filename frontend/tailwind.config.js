/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: "#1e293b",
          low: "#10b981",
          medium: "#f59e0b",
          high: "#ef4444",
        }
      }
    },
  },
  plugins: [],
}