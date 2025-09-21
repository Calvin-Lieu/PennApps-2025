import EdgeAnalysis from "./components/EdgeAnalysis";
import Map from "./components/Map";
import { useMemo, useState, useEffect } from "react";

export default function App() {
  const [date] = useState(() => new Date());
  const edges = useMemo(() => [], []); // No demo edges needed
  const [currentPage, setCurrentPage] = useState<'map' | 'analysis' | 'test'>(() => {
    // Check URL path to determine initial page
    if (window.location.pathname === '/analysis') return 'analysis';
    if (window.location.pathname === '/test') return 'test';
    return 'map';
  });

  // Handle URL changes
  useEffect(() => {
    const handlePopState = () => {
      if (window.location.pathname === '/analysis') {
        setCurrentPage('analysis');
      } else if (window.location.pathname === '/test') {
        setCurrentPage('test');
      } else {
        setCurrentPage('map');
      }
    };
    
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Update URL when page changes
  const navigateToPage = (page: 'map' | 'analysis' | 'test') => {
    const pathMap = {
      'map': '/',
      'analysis': '/analysis',
      'test': '/test'
    };
    const newPath = pathMap[page];
    window.history.pushState({}, '', newPath);
    setCurrentPage(page);
  };

  if (currentPage === 'analysis') {
    return <EdgeAnalysis onBack={() => navigateToPage('map')} />;
  }

  return <Map />;
}
