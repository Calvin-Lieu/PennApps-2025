# Shadenav Deployment Guide

## 🚀 Production Branch Setup

This branch contains production-ready configurations for hosting Shadenav online.

## 📋 Pre-Deployment Checklist

### 1. Environment Variables
Update `.env.production` with your actual values:
```bash
VITE_API_BASE_URL=https://your-backend-domain.com
VITE_SHADEMAP_KEY=your-actual-shademap-api-key
VITE_APP_TITLE=Shadenav
```

### 2. Backend Deployment
- Deploy your FastAPI backend to a cloud service (Railway, Render, Heroku, etc.)
- Ensure CORS is configured to allow your frontend domain
- Test all API endpoints are working

### 3. Frontend Build
```bash
cd frontend
npm install
npm run build:prod
```

### 4. Hosting Options

#### Option A: Vercel (Recommended)
1. Connect your GitHub repository
2. Set build command: `npm run build:prod`
3. Set output directory: `dist`
4. Add environment variables in Vercel dashboard

#### Option B: Netlify
1. Connect your GitHub repository
2. Set build command: `npm run build:prod`
3. Set publish directory: `dist`
4. Add environment variables in Netlify dashboard

#### Option C: GitHub Pages
1. Enable GitHub Pages in repository settings
2. Use GitHub Actions to build and deploy
3. Set environment variables as repository secrets

## 🔧 Configuration

### API Endpoints
- Tree shadows: `GET /tree_shadows`
- Pathfinding: `POST /shortest_path_shade`
- Regular pathfinding: `POST /shortest_path`

### Required Environment Variables
- `VITE_API_BASE_URL`: Your backend API URL
- `VITE_SHADEMAP_KEY`: ShadeMap API key for building shadows

## 🐛 Troubleshooting

### Common Issues
1. **CORS errors**: Ensure backend allows your frontend domain
2. **API key issues**: Verify ShadeMap API key is valid
3. **Build failures**: Check all environment variables are set
4. **Geolocation not working**: Ensure site is served over HTTPS

### Testing
1. Test locally with production build: `npm run preview:prod`
2. Verify all API calls work with production backend
3. Test on mobile devices for geolocation

## 📱 Mobile Considerations
- HTTPS is required for geolocation API
- Touch interactions work on mobile
- Responsive design maintained

## 🔒 Security Notes
- Never commit real API keys to repository
- Use environment variables for all sensitive data
- Enable HTTPS in production
