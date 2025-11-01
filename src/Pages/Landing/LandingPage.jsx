import React from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom"
import "./LandingPage.css";

export default function LandingPage() {
  return (
    <div className="h-screen flex flex-col justify-between text-center background bg-cover bg-center relative overflow-hidden">
      <div className="absolute inset-0 bg-white/70 backdrop-blur-sm"></div>

      {/* Navbar */}
      <nav className="relative z-10 flex items-center justify-between px-5 md:px-8 py-4 shadow-lg">
        <Link to="/" className="text:xl md:text-3xl font-extrabold text-blue-700 tracking-wide">
          Ota Health AI
        </Link>
        <div className="space-x-4">
          <Link to="/login" className="px-6 py-2 rounded-full border-2 border-blue-600 text-blue-600 font-semibold hover:bg-blue-600 hover:text-white transition duration-500 cursor-pointer">
            Login
          </Link>
          <Link to="/register" className="px-6 py-2 rounded-full bg-blue-600 text-white font-semibold hover:bg-blue-700 transition duration-500 cursor-pointer">
            SignUp
          </Link>
        </div>
      </nav>

      <motion.section
        className="relative z-10 flex flex-col items-center justify-center flex-grow px-4"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: "easeOut" }}
      >
        <motion.div
          className="bg-blue-100 border border-blue-300 text-blue-700 px-6 py-2 rounded-full text-sm font-semibold mb-6"
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.6 }}
        >
          Healthcare AI Made Easy
        </motion.div>

        <motion.h2
          className="text-5xl sm:text-4xl font-extrabold text-gray-900 mb-5"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.8 }}
        >
          Nwankwo-Emmanuel-Ota
        </motion.h2>

        <motion.h3
          className="text-2xl sm:text-3xl font-bold text-blue-700 mb-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.8 }}
        >
          AI Powered Health Checker
        </motion.h3>

        <motion.p
          className="max-w-md text-gray-700 font-semibold text-base mb-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 0.8 }}
        >
          A modern platform that uses Artificial Intelligence to make healthcare
          smarter, faster, and more accessible for everyone.
        </motion.p>

        <motion.p
          className="text-blue-900 font-bold text-xl"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.3, duration: 0.8 }}
        >
          Thank you, Doctor Titi 💙
        </motion.p>
      </motion.section>

      {/* Footer */}
      <footer className="relative z-10 text-center py-4 text-sm font-semibold text-gray-600">
        © {new Date().getFullYear()} Ota Health AI — Healthcare made smarter
      </footer>
    </div>
  );
}
