import { NavLink as RouterNavLink } from "react-router-dom";

const NavLink = ({ to, children }) => (
  <RouterNavLink to={to}>{children}</RouterNavLink>
);

export default NavLink;
