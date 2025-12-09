// src/components/Navbar.jsx
import React from "react";
import { NavLink } from "react-router-dom";

export default function Navbar() {
  const active = ({ isActive }) => isActive ? "nav-link active" : "nav-link";

  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container container-card">
        <NavLink className="navbar-brand" to="/">CreditPathAI</NavLink>
        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navMain">
          <span className="navbar-toggler-icon" />
        </button>

        <div className="collapse navbar-collapse" id="navMain">
          <ul className="navbar-nav ms-auto">
            <li className="nav-item"><NavLink to="/" className={active}>Home</NavLink></li>
            <li className="nav-item"><NavLink to="/predict" className={active}>Predict</NavLink></li>
            <li className="nav-item"><NavLink to="/dashboard" className={active}>Dashboard</NavLink></li>
            <li className="nav-item"><NavLink to="/logs" className={active}>Logs</NavLink></li>
          </ul>
        </div>
      </div>
    </nav>
  );
}