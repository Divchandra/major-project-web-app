import React, { useState } from 'react';

function Upload() {
  const [file, setFile] = useState(null);
  const [segmentedImage, setSegmentedImage] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [leftBreastImage, setLeftBreastImage] = useState(null);
  const [rightBreastImage, setRightBreastImage] = useState(null);
  const [result, setResult] = useState(null);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleUpload = async (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setIsSubmitted(false); // Reset results before submitting
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      setResult(data);
      setSegmentedImage(data.segmented_image_url);
      setOriginalImage(data.original_image_url);
      setLeftBreastImage(data.left_breast_image_url);
      setRightBreastImage(data.right_breast_image_url);
      setIsSubmitted(true); // Mark as submitted
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Upload failed. Check the backend.");
    }
  };

  return (
    <div className="p-6 bg-white rounded-2xl shadow-md">
      <h2 className="text-2xl font-semibold text-pink-600 mb-4">Upload Breast Scan Image</h2>

      {/* Upload input */}
      <input
        type="file"
        onChange={handleUpload}
        accept="image/*"
        className="border border-gray-300 rounded p-2 mb-4"
      />
      
      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        className="bg-blue-500 text-white p-2 rounded-md"
      >
        Submit
      </button>

      {/* Display Selected File */}
      {file && !isSubmitted && (
        <p className="mt-2 text-green-600">Selected file: {file.name}</p>
      )}

      {/* Display Classification Results */}
      {result && isSubmitted && (
        <div className="mt-4 text-blue-700">
          <p><strong>Left Breast:</strong> {result.left}</p>
          <p><strong>Right Breast:</strong> {result.right}</p>
        </div>
      )}

      {/* Display the Images */}
      <div className="mt-6">
        {originalImage && isSubmitted && (
          <div className="image-preview">
            <h3 className="text-xl font-semibold text-gray-700">Original Image:</h3>
            <img
              src={originalImage}
              alt="Uploaded"
              className="w-full max-w-sm rounded shadow-md"
              style={{ maxWidth: '100%', height: 'auto' }}
            />
          </div>
        )}

        {segmentedImage && isSubmitted && (
          <div className="image-preview mt-4">
            <h3 className="text-xl font-semibold text-gray-700">Segmented Image:</h3>
            <img
              src={segmentedImage}
              alt="Segmented"
              className="w-full max-w-sm rounded shadow-md"
              style={{ maxWidth: '100%', height: 'auto' }}
            />
          </div>
        )}

        {/* Display Left and Right Breast Images Side by Side */}
        {leftBreastImage && rightBreastImage && isSubmitted && (
          <div className="images-container mt-4">
            <div className="image-preview left-image">
              <h3 className="text-xl font-semibold text-gray-700">Left Breast Image:</h3>
              <img
                src={leftBreastImage}
                alt="Left Breast"
                className="left-breast-image"
              />
            </div>
            <div className="image-preview right-image">
              <h3 className="text-xl font-semibold text-gray-700">Right Breast Image:</h3>
              <img
                src={rightBreastImage}
                alt="Right Breast"
                className="right-breast-image"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Upload;
