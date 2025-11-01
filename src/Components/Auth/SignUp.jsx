import React, { useState } from "react";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { Eye, EyeOff } from "lucide-react";
import Background from "../../assets/background.jpg";
import { Link, useNavigate } from "react-router-dom";

const SignUp = () => {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    position: "",
    department: "",
  });

  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const validateForm = () => {
    const { name, email, password, position, department } = formData;

    if (!name || !email || !password || !position || !department) {
      toast.error("Please fill out all fields before submitting!");
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      toast.error("Please enter a valid email address!");
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;
    setLoading(true);

    try {
      const res = await fetch("https://health-result-project.onrender.com/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await res.json();

      if (!res.ok) {
        if (data.detail && Array.isArray(data.detail)) {
          const msg = data.detail[0]?.msg || "Invalid input!";
          toast.error(msg);
        }
        else if (
          data.message?.toLowerCase().includes("email") ||
          JSON.stringify(data).toLowerCase().includes("email")
        ) {
          toast.warning("This email is already registered!");
        } else {
          toast.error(data.message || "Registration failed!");
        }
        return;
      }

      toast.success("Account created successfully!");
      setTimeout(() => navigate("/login"), 2000);

      setFormData({
        name: "",
        email: "",
        password: "",
        position: "",
        department: "",
      });
    } catch (error) {
      toast.error(error.message || "Something went wrong!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-full">
      {/* Left Side Image */}
      <div className="hidden md:flex w-1/2">
        <img
          src={Background}
          alt="Background"
          className="object-cover w-full h-full"
        />
      </div>

      {/* Right Side Form */}
      <div className="w-full md:w-1/2 bg-[#3592c9] flex flex-col justify-center items-center p-8 text-white">
        <div className="max-w-md w-full animate-fadeIn">
          <h1 className="text-3xl font-bold mb-2 animate-slideUp">
            Create an account
          </h1>
          <p className="text-sm mb-8 animate-slideUp">
            Already have an account?{" "}
            <Link
              to="/login"
              className="underline font-semibold text-gray-200 hover:text-white"
            >
              Login
            </Link>
          </p>

          <form className="space-y-5" onSubmit={handleSubmit}>
            {["name", "email", "position", "department", "password"].map(
              (field) => (
                <div key={field} className="animate-slideUp">
                  <label className="block text-sm font-medium mb-2 capitalize">
                    {field}
                  </label>
                  <div className="relative">
                    <input
                      type={
                        field === "password"
                          ? showPassword
                            ? "text"
                            : "password"
                          : field === "email"
                          ? "email"
                          : "text"
                      }
                      name={field}
                      value={formData[field]}
                      onChange={handleChange}
                      placeholder={`Enter your ${field}`}
                      className="w-full px-4 py-2 rounded-md text-gray-800 border-2 border-[#2a7aae] outline-none focus:ring-2 focus:ring-[#5bb6e6]"
                      required
                    />
                    {field === "password" && (
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-2.5 text-gray-600 hover:text-gray-800"
                      >
                        {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                      </button>
                    )}
                  </div>
                </div>
              )
            )}

            <button
              type="submit"
              disabled={loading}
              className={`w-full bg-white text-[#2857e8] font-bold py-2 cursor-pointer rounded-md transform transition-all duration-300 hover:scale-105 ${
                loading ? "opacity-70 cursor-not-allowed" : ""
              }`}
            >
              {loading ? "Signing Up..." : "Sign Up"}
            </button>
          </form>
        </div>
      </div>

      <ToastContainer position="top-right" autoClose={3000} hideProgressBar />
    </div>
  );
};

export default SignUp;
