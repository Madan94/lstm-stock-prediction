/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        leaf: {
          DEFAULT: '#22c55e', // Leaf green
          light: '#4ade80',
          dark: '#16a34a',
          lighter: '#86efac',
        },
      },
    },
  },
  plugins: [],
}



