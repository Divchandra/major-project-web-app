import React from 'react';

function Methodology() {
  return (
    <div className="p-6 bg-pink-50 rounded-2xl shadow-md">
      <h2 className="text-2xl font-bold text-pink-700 mb-4">Methodology</h2>
      <p className="text-1-gray-700 text-lg mb-6">
      The AI-based breast cancer detection system processes thermal images by first denoising and segmenting the breast region. It then detects the Region of Interest (ROI), separates the left and right breasts, and classifies each to determine if the tumor is benign or malignant. This streamlined workflow enables accurate tumor detection
      </p>
      <div className="w-1 bg-pink-500 h-full absolute top-full left-1/2 transform -translate-x-1/2"></div>
      

      <div className="flex justify-center items-center space-x-8 mb-8">
        <span className="text-gray-700 text-lg">Start →</span>

        {/* Circle layout for images */}
        <div className="flex items-center justify-center space-x-4">
          {/* Image 1 */}
          <div className="circle-image"  style={{ backgroundImage: "url('/images/image.png')" }}></div>
            <p className="text-center text-sm mt-2">Original Image</p>
            <div class="vl"></div>

          {/* Image 2 */}
          <div className="circle-image" style={{ backgroundImage: "url('/images/C001.jpg')" }}></div>
            <p className="text-center text-sm mt-2">Denoising</p>
            <div class="vl"></div>

          {/* Image 3 */}
          <div className="circle-image" style={{ backgroundImage: "url('/images/image3.jpg')" }}></div>
            <p className="text-center text-sm mt-2">Segmentation</p>
            <div class="vl"></div>
    

          {/* Image 4 */}
          <div className="circle-image" style={{ backgroundImage: "url('/images/one_breast.png')" }}></div>
            <p className="text-center text-sm mt-2">Classification</p>
            <div class="vl"></div>
    
        </div>

        <span className="text-2-gray-700 text-lg">Final Result :Benign or Malignant</span>
      </div>
    </div>
  );
}

export default Methodology;
