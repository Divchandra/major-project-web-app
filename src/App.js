import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import About from './components/About';
import Upload from './components/Upload';
import Methodology from './components/Methodology';
import Footer from './components/footer';

// Typewriter effect component
const TypewriterEffect = ({ text, speed = 100 }) => {
  const [displayedText, setDisplayedText] = useState('');
  
  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      setDisplayedText((prev) => prev + (text[index] || ''));
      index += 1;
      if (index === text.length) {
        clearInterval(interval); // Stop the interval once all text is displayed
      }
    }, speed);

    return () => clearInterval(interval); // Cleanup the interval when component unmounts
  }, [text, speed]);

  return <span>{displayedText}</span>;
};


function App() {
  return (
    <div className="min-h-screen bg-gradient-to-r from-pink-50 to-pink-100 font-sans">
      <Navbar />
      <header className="relative h-[500px] bg-pink-600 rounded-b-2xl shadow-md overflow-hidden flex">
  {/* Left side: text content */}
  <div className="w-1/2 flex flex-col justify-center pl-12 z-10 text-white">
    <h1 className="text-4xl font-extrabold mb-4">Breast Cancer Detection Portal</h1>
    <p className="text-xl mb-4">AI-powered image analysis for early diagnosis</p>
    <h2 className="text-2xl font-semibold">
      <TypewriterEffect text="A  Collaboration between UTPL X NITK" speed={100} />
    </h2>
  </div>
  

  {/* Right side: image */}
  <div className="w-1/2 h-full relative z-0">
    <img
      src="/images/doctor.jpg"
      alt="Doctor"
      className="w-full h-full object-cover opacity-40"
    />
  </div>
</header>



      

      

      <main className="p-8 space-y-8 max-w-4xl mx-auto">
        <section id="about">
          <About />
        </section>

        <section id="upload">
          <Upload />
        </section>

        <section id="methodology">
          <Methodology />
        </section>
      </main>

      <Footer />
    </div>
  );
}

export default App;
