FROM node:22-bookworm-slim

WORKDIR /app

# Install from the lockfile before copying source so Docker can reuse this
# layer while the interface is being iterated on.
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund

COPY . ./

EXPOSE 3001

# Vinext provides the development-time reverse proxy from /chat to ATLAS.
CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0", "--port", "3001"]
