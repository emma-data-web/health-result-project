import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import "./Report.css";

const Report = () => {
  const navigate = useNavigate();
  const [activeForm, setActiveForm] = useState("malaria");
  const [showPopup, setShowPopup] = useState(false);
  const [popupMessage, setPopupMessage] = useState("");

  const [malariaData, setMalariaData] = useState({
    Age: "",
    Body_Temperature: "",
    Hemoglobin: "",
    RBC_Count: "",
    Platelet_Count: "",
    Has_Fever: false,
    Has_Chills: false,
    Has_Vomiting: false,
    Rainy_Season: false,
  });

  const [diabetesData, setDiabetesData] = useState({
    Pregnancies: "",
    Glucose: "",
    BloodPressure: "",
    SkinThickness: "",
    Insulin: "",
    BMI: "",
    DiabetesPedigreeFunction: "",
    Age: "",
  });

  const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    navigate("/login");
  };

  const handleMalariaChange = (e) => {
    const { name, value, type, checked } = e.target;
    setMalariaData({
      ...malariaData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const handleDiabetesChange = (e) => {
    const { name, value } = e.target;
    setDiabetesData({
      ...diabetesData,
      [name]: value,
    });
  };

  const handleMalariaSubmit = async (e) => {
    e.preventDefault();

    const {
      Age,
      Body_Temperature,
      Hemoglobin,
      RBC_Count,
      Platelet_Count,
      Has_Fever,
      Has_Chills,
      Has_Vomiting,
      Rainy_Season,
    } = malariaData;

    if (
      !Age ||
      !Body_Temperature ||
      !Hemoglobin ||
      !RBC_Count ||
      !Platelet_Count
    ) {
      setPopupMessage("Please fill all required fields before submitting!");
      setShowPopup(true);
      return;
    }

    const payload = {
      Age: Number(Age),
      Body_Temperature: Number(Body_Temperature),
      Hemoglobin: Number(Hemoglobin),
      RBC_Count: Number(RBC_Count),
      Platelet_Count: Number(Platelet_Count),
      Has_Fever: Has_Fever ? 1 : 0,
      Has_Chills: Has_Chills ? 1 : 0,
      Has_Vomiting: Has_Vomiting ? 1 : 0,
      Rainy_Season: Rainy_Season ? 1 : 0,
    };

    try {
      const res = await fetch(
        "https://health-result-project.onrender.com/malpredict",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.detail?.[0]?.msg || "Server Error!");
      }

      setPopupMessage(`
        <strong style="font-size: 20px; color: ${
          data?.Result === 1 ? "#dc2626" : "#2563eb"
        };">
          Malaria Result: ${data?.Result === 1 ? "Positive" : "Negative"}
        </strong><br/><br />
        <span style="font-size: 16px; color: #374151;">
          Confidence:${
            data?.confidence || "N/A"
          }
        </span>
      `);

      setShowPopup(true);

      setMalariaData({
        Age: "",
        Body_Temperature: "",
        Hemoglobin: "",
        RBC_Count: "",
        Platelet_Count: "",
        Has_Fever: false,
        Has_Chills: false,
        Has_Vomiting: false,
        Rainy_Season: false,
      });
    } catch (error) {
      setPopupMessage(error.message || "Something went wrong!");
      setShowPopup(true);
    }
  };

  const handleDiabetesSubmit = async (e) => {
    e.preventDefault();

    const {
      Pregnancies,
      Glucose,
      BloodPressure,
      SkinThickness,
      Insulin,
      BMI,
      DiabetesPedigreeFunction,
      Age,
    } = diabetesData;

    if (
      !Pregnancies ||
      !Glucose ||
      !BloodPressure ||
      !SkinThickness ||
      !Insulin ||
      !BMI ||
      !DiabetesPedigreeFunction ||
      !Age
    ) {
      setPopupMessage("Please fill all required fields before submitting!");
      setShowPopup(true);
      return;
    }

    const payload = {
      Pregnancies: Number(Pregnancies),
      Glucose: Number(Glucose),
      BloodPressure: Number(BloodPressure),
      SkinThickness: Number(SkinThickness),
      Insulin: Number(Insulin),
      BMI: Number(BMI),
      DiabetesPedigreeFunction: Number(DiabetesPedigreeFunction),
      Age: Number(Age),
    };

    try {
      const res = await fetch(
        "https://health-result-project.onrender.com/diapredict",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.detail?.[0]?.msg || "Server Error!");
      }

      setPopupMessage(`
        <strong style="font-size: 20px; color: ${
          data?.Outcome === 1 ? '#dc2626' : '#2563eb'
        };">
          Diabetes Result: ${data?.Outcome === 1 ? "Positive" : "Negative"}
        </strong>
      `);
      
      setShowPopup(true);

      setDiabetesData({
        Pregnancies: "",
        Glucose: "",
        BloodPressure: "",
        SkinThickness: "",
        Insulin: "",
        BMI: "",
        DiabetesPedigreeFunction: "",
        Age: "",
      });
    } catch (error) {
      setPopupMessage(error.message || "Something went wrong!");
      setShowPopup(true);
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-between text-center background bg-cover bg-center relative overflow-auto">
      <div className="absolute inset-0 bg-white/20 pointer-events-none"></div>

      <nav className="relative z-10 flex items-center justify-between px-5 md:px-8 py-4 shadow-lg">
        <Link
          to="/"
          className="text:xl md:text-3xl font-extrabold text-blue-700 tracking-wide"
        >
          Ota Health AI
        </Link>
        <button
          onClick={handleLogout}
          className="px-6 py-2 rounded-full bg-blue-600 text-white font-semibold hover:bg-blue-700 transition duration-500 cursor-pointer"
        >
          Logout
        </button>
      </nav>

      <div className="flex flex-col items-center py-8 space-y-6 relative z-10">
        <h2 className="text-2xl font-bold text-gray-800">Choose AI Action</h2>
        <div className="flex space-x-4">
          <button
            onClick={() => setActiveForm("malaria")}
            className={`px-6 py-2 rounded-full font-semibold transition cursor-pointer duration-300 ${
              activeForm === "malaria"
                ? "bg-blue-600 text-white"
                : "bg-gray-200 text-gray-700 hover:bg-gray-300"
            }`}
          >
            Malaria
          </button>
          <button
            onClick={() => setActiveForm("diabetes")}
            className={`px-6 py-2 rounded-full font-semibold transition duration-300 cursor-pointer ${
              activeForm === "diabetes"
                ? "bg-blue-600 text-white"
                : "bg-gray-200 text-gray-700 hover:bg-gray-300"
            }`}
          >
            Diabetes
          </button>
        </div>
      </div>

      <div className="flex justify-center mb-10 relative z-10">
        {activeForm === "malaria" && (
          <form
            onSubmit={handleMalariaSubmit}
            className="bg-white p-8 rounded-2xl shadow-lg w-full max-w-2xl space-y-6"
          >
            <h2 className="text-xl font-bold text-blue-700 text-center">
              Send Malaria Report
            </h2>

            <div className="grid grid-cols-2 gap-4">
              {[
                "Age",
                "Body_Temperature",
                "Hemoglobin",
                "RBC_Count",
                "Platelet_Count",
              ].map((key) => (
                <label key={key} className="text-left">
                  <span className="block text-sm font-medium text-gray-700 mb-1">
                    {key.replace("_", " ")}
                  </span>
                  <input
                    type="number"
                    name={key}
                    placeholder={key.replace("_", " ")}
                    value={malariaData[key]}
                    onChange={handleMalariaChange}
                    required
                    className="border rounded-md px-3 py-2 w-full"
                  />
                </label>
              ))}
            </div>

            <div className="flex flex-wrap gap-4 justify-center mt-4">
              {["Has_Fever", "Has_Chills", "Has_Vomiting", "Rainy_Season"].map(
                (item) => (
                  <label key={item} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      name={item}
                      checked={malariaData[item]}
                      onChange={handleMalariaChange}
                    />
                    <span className="capitalize">{item.replace("_", " ")}</span>
                  </label>
                )
              )}
            </div>

            <button
              type="submit"
              className="w-full bg-blue-600 text-white font-semibold py-2 rounded-md hover:bg-blue-700 transition duration-300"
            >
              Send Report
            </button>
          </form>
        )}

        {activeForm === "diabetes" && (
          <form
            onSubmit={handleDiabetesSubmit}
            className="bg-white p-8 rounded-2xl shadow-lg w-full max-w-2xl space-y-6"
          >
            <h2 className="text-xl font-bold text-blue-700 text-center">
              Send Diabetes Report
            </h2>

            <div className="grid grid-cols-2 gap-4">
              {Object.keys(diabetesData).map((key) => (
                <label key={key} className="text-left">
                  <span className="block text-sm font-medium text-gray-700 mb-1">
                    {key.replace("_", " ")}
                  </span>
                  <input
                    type="number"
                    name={key}
                    placeholder={key.replace("_", " ")}
                    value={diabetesData[key]}
                    onChange={handleDiabetesChange}
                    required
                    className="border rounded-md px-3 py-2 w-full"
                  />
                </label>
              ))}
            </div>

            <button
              type="submit"
              className="w-full bg-blue-600 text-white font-semibold py-2 rounded-md hover:bg-blue-700 transition duration-300 cursor-pointer"
            >
              Send Report
            </button>
          </form>
        )}
      </div>

      {showPopup && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-50 transition-opacity duration-300">
          <div className="bg-gradient-to-br from-white via-blue-50 to-white rounded-2xl shadow-2xl p-10 w-11/12 max-w-md text-center animate-fadeIn">
            <h3 className="text-3xl font-extrabold mb-4 text-blue-700 drop-shadow-sm">
              AI Health Report
            </h3>
            <p
              className="text-lg leading-relaxed mb-8 font-medium border border-blue-100 rounded-lg p-4 bg-blue-50/30"
              dangerouslySetInnerHTML={{ __html: popupMessage }}
            />

            <button
              onClick={() => setShowPopup(false)}
              className="px-8 py-3 bg-blue-600 text-white font-semibold rounded-full shadow-md hover:bg-blue-700 active:scale-95 transition-all duration-300 cursor-pointer"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Report;
