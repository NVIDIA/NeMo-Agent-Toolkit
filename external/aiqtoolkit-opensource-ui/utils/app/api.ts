import { nextEndPoints } from "./const";

export const getEndpoint = ({ service = 'chat'}) => {
  return nextEndPoints[service]
};

