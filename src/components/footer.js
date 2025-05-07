import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faInstagram, faLinkedin, faTwitter } from '@fortawesome/free-brands-svg-icons';

function Footer() {
  return (
    <footer className="bg-pink-700 text-white py-10 px-6 rounded-t-2xl w-full">
      <div className="max-w-screen-xl mx-auto text-center">
        <h2 className="text-2xl font-bold mb-4">Connect With Us</h2>

        {/* Social Media Icons */}
        <div className="flex justify-center space-x-6 mb-8 text-2xl">
          <a href="https://www.instagram.com/" target="_blank" rel="noopener noreferrer">
            <FontAwesomeIcon icon={faInstagram} className="hover:text-pink-300" />
          </a>
          <a href="https://www.linkedin.com/" target="_blank" rel="noopener noreferrer">
            <FontAwesomeIcon icon={faLinkedin} className="hover:text-pink-300" />
          </a>
          <a href="https://www.twitter.com/" target="_blank" rel="noopener noreferrer">
            <FontAwesomeIcon icon={faTwitter} className="hover:text-pink-300" />
          </a>
        </div>

        {/* Team Photos with circular cropping */}
        <div className="flex justify-center gap-12 mt-4 flex-wrap">
  <div className="flex flex-col items-center">
    <div
      className="circle-image-foot"
      style={{ backgroundImage: "url('/images/div.jpg')" }}
    ></div>
    
  </div>
  
  <div className="flex flex-col items-center">
    <div
      className="circle-image-foot2"
      style={{ backgroundImage: "url('/images/vand.jpg')" }}
    ></div>
   {/* Adjust mt-4 as needed */}
    
  </div>
  
</div>
<span>
  Divyangana Chandra <span style={{ marginLeft: '60px' }}>Vandana S</span>
</span>


        <p className="mt-8 text-sm">© 2025 Breast Cancer Detection Team. All rights reserved.</p>
      </div>
    </footer>
  );
}

export default Footer;
