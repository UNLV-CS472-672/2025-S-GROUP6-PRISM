import type { NextConfig } from "next";

const nextConfig: NextConfig = {
    trailingSlash: true,
    assetPrefix: process.env.BASE_PATH || '',
};

export default nextConfig;
