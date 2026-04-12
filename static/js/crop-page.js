document.addEventListener("DOMContentLoaded", function () {
    const revealItems = document.querySelectorAll(".reveal");
    const currentBtn = document.getElementById("currentLocationBtn");
    const manualBtn = document.getElementById("manualLocationBtn");

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("show");
                }
            });
        },
        {
            threshold: 0.12
        }
    );

    revealItems.forEach((item) => observer.observe(item));

    if (currentBtn && manualBtn) {
        currentBtn.addEventListener("click", function () {
            currentBtn.classList.add("active");
            manualBtn.classList.remove("active");
        });

        manualBtn.addEventListener("click", function () {
            manualBtn.classList.add("active");
            currentBtn.classList.remove("active");
        });
    }
});