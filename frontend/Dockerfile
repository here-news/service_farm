FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source
COPY . .

# Default: build to ../static (mount point)
CMD ["npm", "run", "build"]
