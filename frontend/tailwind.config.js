/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        trading: {
          green: '#22c55e',
          'green-light': '#4ade80',
          'green-dark': '#16a34a',
          black: '#000000',
          'black-light': '#1a1a1a',
          'black-lighter': '#2a2a2a',
          white: '#ffffff',
        },
        leaf: {
          DEFAULT: '#22c55e',
          light: '#4ade80',
          dark: '#16a34a',
          lighter: '#86efac',
        },
      },
    },
  },
  plugins: [],
}



