import React, { useState } from "react";
import Background from "../../assets/background.jpg";
import { Link, useNavigate } from "react-router-dom";
import { Eye, EyeOff } from "lucide-react";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const Login = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const isValidEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!isValidEmail(formData.email)) {
      toast.error("Please enter a valid email address!");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch("https://health-result-project.onrender.com/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await res.json();

      if (!res.ok) {
        let errorMsg = "Invalid email or password!";

        if (data?.message) errorMsg = data.message;
        else if (data?.detail && Array.isArray(data.detail)) {
          errorMsg = data.detail[0]?.msg || errorMsg;
        } else if (typeof data?.detail === "string") {
          errorMsg = data.detail;
        } else if (data?.error) {
          errorMsg = data.error;
        }

        throw new Error(errorMsg);
      }

      toast.success("Login successful!");
      localStorage.setItem("isLoggedIn", "true");
      setFormData({ email: "", password: "" });
      setTimeout(() => navigate("/report"), 1000);
    } catch (error) {
      toast.error(error.message || "Login failed. Please try again.");
      localStorage.removeItem("isLoggedIn");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-full">
      <div className="hidden md:flex w-1/2">
        <img src={Background} alt="Background" className="object-cover w-full h-full" />
      </div>

      <div className="w-full md:w-1/2 bg-[#3592c9] flex flex-col justify-center items-center p-8 text-white">
        <div className="max-w-md w-full">
          <h1 className="text-3xl font-bold mb-2">Login</h1>
          <p className="text-sm mb-8">
            Don’t have an account?{" "}
            <Link to="/register" className="underline font-semibold text-gray-200 hover:text-white">
              Sign Up
            </Link>
          </p>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-medium mb-2">Email</label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="Enter your email"
                className="w-full px-4 py-2 rounded-md text-gray-800 border-2 border-[#2a7aae] outline-none"
                required
              />
            </div>

            <div className="relative">
              <label className="block text-sm font-medium mb-2">Password</label>
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Enter your password"
                className="w-full px-4 py-2 rounded-md text-gray-800 border-2 border-[#2a7aae] outline-none focus:ring-2 focus:ring-[#5bb6e6]"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-10 text-gray-600 hover:text-gray-800"
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`w-full bg-white text-[#2857e8] font-bold py-2 rounded-md cursor-pointer transform transition-all duration-300 hover:scale-105 ${
                loading ? "opacity-70 cursor-not-allowed" : ""
              }`}
            >
              {loading ? "Logging in..." : "Login"}
            </button>
          </form>
        </div>
      </div>

      <ToastContainer position="top-right" autoClose={3000} hideProgressBar />
    </div>
  );
};

export default Login;
