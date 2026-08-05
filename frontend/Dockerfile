# --- build -----------------------------------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

# Copy manifests first so `npm ci` is cached until dependencies actually change.
COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY tsconfig.json tsconfig.node.json vite.config.ts index.html ./
COPY src ./src

# `npm run build` typechecks before bundling, so a type error fails the image
# build rather than shipping.
RUN npm run build

# --- serve -----------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=4s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
