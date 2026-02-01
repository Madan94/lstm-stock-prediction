import type { AppProps } from 'next/app';
import '../styles/globals.css';
import Navbar from '../components/Layout/Navbar';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <Component {...pageProps} />
    </div>
  );
}



