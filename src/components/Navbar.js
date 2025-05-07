import React from 'react';

function Navbar() {
  const scrollToSection = (id) => {
    const section = document.getElementById(id);
    if (section) {
      section.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <nav className="sticky top-0 z-50 bg-pink-700 text-white px-6 py-4 shadow-lg flex justify-center space-x-6">
      <button onClick={() => scrollToSection('about')} className="hover:text-pink-200">About</button>
      <button onClick={() => scrollToSection('upload')} className="hover:text-pink-200">Upload</button>
      <button onClick={() => scrollToSection('methodology')} className="hover:text-pink-200">Methodology</button>
    </nav>
  );
}

export default Navbar;
