const STATIC_CACHE = "harvestify-static-v1";
const PAGE_CACHE = "harvestify-pages-v1";

const APP_SHELL = [
    "/",
    "/?lang=en",
    "/?lang=mr",
    "/crop-recommend",
    "/crop-recommend?lang=en",
    "/crop-recommend?lang=mr",
    "/fertilizer",
    "/fertilizer?lang=en",
    "/fertilizer?lang=mr",
    "/static/css/bootstrap.css",
    "/static/css/style.css",
    "/static/css/font-awesome.min.css",
    "/static/css/custom-ui.css",
    "/static/scripts/cities.js",
    "/static/images/favicon.ico"
];

self.addEventListener("install", (event) => {
    event.waitUntil(
        caches.open(STATIC_CACHE).then((cache) => cache.addAll(APP_SHELL))
    );
    self.skipWaiting();
});

self.addEventListener("activate", (event) => {
    event.waitUntil(
        caches.keys().then((keys) => Promise.all(
            keys
                .filter((key) => ![STATIC_CACHE, PAGE_CACHE].includes(key))
                .map((key) => caches.delete(key))
        ))
    );
    self.clients.claim();
});

self.addEventListener("fetch", (event) => {
    const request = event.request;

    if (request.method !== "GET") {
        return;
    }

    const url = new URL(request.url);

    if (request.mode === "navigate") {
        event.respondWith(
            fetch(request)
                .then((response) => {
                    const copy = response.clone();
                    caches.open(PAGE_CACHE).then((cache) => cache.put(request, copy));
                    return response;
                })
                .catch(() =>
                    caches.match(request).then((cached) => cached || caches.match("/crop-recommend?lang=en"))
                )
        );
        return;
    }

    if (url.origin === self.location.origin) {
        event.respondWith(
            caches.match(request).then((cached) => {
                if (cached) {
                    return cached;
                }

                return fetch(request).then((response) => {
                    const copy = response.clone();
                    caches.open(STATIC_CACHE).then((cache) => cache.put(request, copy));
                    return response;
                });
            })
        );
    }
});
