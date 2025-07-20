/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Clean Professional Palette with White Background
        'enterprise-dark': '#1a1a2e',
        'enterprise-darker': '#16213e',
        'enterprise-accent': '#4f46e5', // Indigo
        'enterprise-secondary': '#7c3aed', // Purple
        'enterprise-success': '#059669', // Emerald
        'enterprise-danger': '#dc2626', // Red
        'enterprise-warning': '#d97706', // Amber
        'enterprise-info': '#0891b2', // Cyan
        'enterprise-surface': '#f8fafc', // Light gray
        'enterprise-border': '#e2e8f0',
        'enterprise-text': '#1e293b', // Dark text
        'enterprise-text-secondary': '#64748b',
        'panel-primary': '#ffffff', // White panels
        'panel-secondary': '#f1f5f9', // Light gray panels
        'panel-accent': '#4f46e5', // Indigo accent
        'panel-success': '#10b981', // Green
        'panel-danger': '#ef4444', // Red
        'panel-warning': '#f59e0b', // Amber
        'panel-info': '#06b6d4', // Cyan
        'enterprise-gradient': {
          'from': '#ffffff',
          'via': '#f8fafc', 
          'to': '#f1f5f9'
        },
        'accent-gradient': {
          'from': '#4f46e5',
          'via': '#6366f1',
          'to': '#7c3aed'
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 6s ease-in-out infinite',
        'sparkle': 'sparkle 1.5s ease-in-out infinite',
        'slide-up': 'slideUp 0.5s ease-out',
        'slide-down': 'slideDown 0.5s ease-out',
        'fade-in': 'fadeIn 0.3s ease-out',
        'scale-in': 'scaleIn 0.3s ease-out',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px #4f46e5' },
          '100%': { boxShadow: '0 0 20px #4f46e5, 0 0 30px #4f46e5' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        sparkle: {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.8', transform: 'scale(1.1)' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.9)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        }
      },
      backgroundImage: {
        'enterprise-gradient': 'linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%)',
        'accent-gradient': 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
        'success-gradient': 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
        'danger-gradient': 'linear-gradient(135deg, #dc2626 0%, #ef4444 100%)',
        'panel-gradient': 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
      }
    },
  },
  plugins: [],
} 