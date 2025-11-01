import { Routes, Route } from "react-router-dom";
import LandingPage from "./Pages/Landing/LandingPage";
import Login from "./Components/Auth/Login";
import SignUp from "./Components/Auth/SignUp";
import Report from "./Pages/Report/Report";
import ProtectedRoute from "./Components/ProtectedRoute";

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<SignUp />} />
      
      <Route
        path="/report"
        element={
          <ProtectedRoute>
            <Report />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default App;
