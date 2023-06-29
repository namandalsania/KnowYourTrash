import React, { useState } from 'react';
import axios from 'axios';
import Dropzone from 'react-dropzone';

const WasteClassification = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);

  const handleImageUpload = async (file) => {
    const formData = new FormData();
    formData.append('image', file[0]);
    const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    console.log(response);
    setResult(response.data);
  };

  return (
<div className="container">
  <h1 className="title">Know Your Trash</h1>
  <div className="dropzone-container">
    {image ? (
      <div className="image-wrapper">
        <img
          src={URL.createObjectURL(image[0])}
          alt="Selected"
          className="image"
        />
    
      </div>
    ) : (
      <div className="dropzone-wrapper">
        <Dropzone onDrop={(acceptedFiles) => setImage(acceptedFiles)}>
          {({ getRootProps, getInputProps }) => (
            <div {...getRootProps()} className="dropzone">
              <input {...getInputProps()} />
              <p>Drag and drop an image here, or click to select an image</p>
            </div>
          )}
        </Dropzone>
      </div>
    )}
  </div>
  {image && (
    <div className="button-container">
    <button className="button submit-button" onClick={() => handleImageUpload(image)}>
      Submit
    </button>
    {image && (
      <button className="button change-image-button" onClick={() => {
                                                      setImage(null);
                                                      setResult(null);
          }}>
             Change Image
        </button>
    )}
  </div>
  )}
  <div className="result-container">
    {result && (
      <div className="result">
        <h2 className="result-title">{result.result}</h2>
      </div>
    )}
  </div>
</div>

  );
};

export default WasteClassification;
