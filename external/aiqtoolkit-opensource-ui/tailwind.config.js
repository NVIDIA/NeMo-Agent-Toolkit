/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      screens: {
        'xs': '320px',  // Extra small screen breakpoint
        'sm': '344px',  // Small screen breakpoint
        'base': '768px',
        'md': '960px',
        'lg': '1440px',
      },
      fontSize: {
        'xs': ['0.6rem', { lineHeight: '1rem' }],    // Extra small screen font size
        'sm': ['0.875rem', { lineHeight: '1.25rem' }], // Small screen font size
        'base': ['0.9rem', { lineHeight: '1.5rem' }],    // Base font size
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],  // Large screen font size
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],   // Extra large screen font size
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0 },
        },
        flicker: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.4' },
        },
        glitch: {
          '0%': { transform: 'translate(0)' },
          '20%': { transform: 'translate(-2px, 2px)' },
          '40%': { transform: 'translate(2px, -2px)' },
          '60%': { transform: 'translate(-2px, -2px)' },
          '80%': { transform: 'translate(2px, 2px)' },
          '100%': { transform: 'translate(0)' },
        },
        ghost: {
          '0%': { opacity: '0' },
          '50%': { opacity: '1' },
          '100%': { opacity: '0' },
        },
        flash: {
          '0%': { backgroundColor: 'rgba(255, 255, 255, 0)' },
          '50%': { backgroundColor: 'rgba(255, 255, 255, 0.5)' },
          '100%': { backgroundColor: 'rgba(255, 255, 255, 0)' },
        },
        crack1: {
          '0%': {
            transform: 'scale(1)',
            opacity: '1',
          },
          '20%': {
            transform: 'scale(1.05)',
            opacity: '0.8',
          },
          '40%': {
            transform: 'scale(1)',
            opacity: '0.6',
          },
          '60%': {
            transform: 'scale(0.95)',
            opacity: '0.4',
          },
          '80%': {
            transform: 'scale(1)',
            opacity: '0.2',
          },
          '100%': {
            transform: 'scale(1)',
            opacity: '0',
          },
        },
        darken: {
          '0%': { backgroundColor: 'rgba(0, 0, 0, 0)' },
          '100%': { backgroundColor: 'rgba(0, 0, 0, 0.7)' },
        },
        crack: {
          '0%': { backgroundSize: '100%', opacity: '1' },
          '50%': { backgroundSize: '120%', opacity: '1' },
          '100%': { backgroundSize: '100%', opacity: '0' },
        },
        loadingBar: {
          '0%': { transform: 'translateX(-100%)' },
          '50%': { transform: 'translateX(0%)' },
          '100%': { transform: 'translateX(100%)' },
        }
      },
      animation: {
        blink: 'blink 1s step-start infinite',
        flicker: 'flicker 1.5s infinite',
        glitch: 'glitch 1s infinite',
        ghost: 'ghost 3s ease-in-out infinite',
        flash: 'flash 0.5s ease-in-out', // Add your flash animation here
        crack: 'crack 0.6s ease-in-out forwards',
        darken: 'darken 1s forwards',
        loadingBar: 'loadingBar 2s ease-in-out infinite',
      },
    },
  },

  variants: {
    extend: {
      visibility: ['group-hover'],
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
