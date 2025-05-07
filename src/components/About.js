import React from 'react';

function About() {
  return (
    <div className="p-8 bg-pink-100 bg-[url('/public/images/bg_image.png')] bg-cover">
      <div className="bg-white bg-opacity-80 p-6 rounded-xl flex flex-col md:flex-row items-center justify-between gap-8">
        <div className="flex-1">
          <h2 className="text-4xl font-bold text-pink-700 mb-4">About the Project</h2>
          <p className="text-1-gray-700 text-lg mb-4">
            <strong>Breast Thermography Classification App</strong> is a mobile application aimed at assisting in the early detection and classification of breast abnormalities using thermographic imaging. The app leverages image processing and machine learning techniques to provide a non-invasive, AI-supported tool for breast cancer diagnostics.
          </p>

          <h3 className="text-2xl font-semibold text-pink-600 mt-6 mb-2">🎯 General Objective</h3>
          <p className="text-1-gray-700 text-lg mb-4">
            Develop a graphical mobile interface for breast thermal image denoising, segmentation, and classification to determine BI-RADS classes.
          </p>

          <h3 className="text-2xl font-semibold text-pink-600 mt-6 mb-2">🧠 Specific Objectives</h3>
          <ul className="list-disc list-inside text-1-gray-700 text-lg mb-4">
            <li>Design and implement a preprocessing module to reduce noise in thermographic images.</li>
            <li>Segment regions of interest (ROIs) in the images for focused analysis.</li>
            <li>Classify ROI thermographic images into benign or malignant categories.</li>
            <li>Evaluate model performance through statistical analysis.</li>
          </ul>

          <h3 className="text-2xl font-semibold text-pink-600 mt-6 mb-2">🚀 Expected Results</h3>
          <ul className="list-disc list-inside text-1-gray-700 text-lg">
            <li>Efficient algorithms for denoising, segmentation, and classification.</li>
            <li>Filtering system that enhances thermal image quality.</li>
            <li>An integrated app capable of classifying thermographic images with over 90% accuracy.</li>
          </ul>
        </div>
        <div className="flex-1">
          <img className="circle-image1"src="/images/doctor2.jpg" alt="Breast Thermography"  />
        </div>
      </div>
    </div>
  );
}

export default About;