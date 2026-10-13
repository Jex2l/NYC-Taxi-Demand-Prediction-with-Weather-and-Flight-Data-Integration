// NYC Taxi Demand Predictor — presentation-only UI behavior.
// The form still does a normal POST to the Flask route (app.py is untouched);
// this just gives the button a loading state while the page round-trips.
document.addEventListener('DOMContentLoaded', function () {
    var form = document.getElementById('predict-form');
    var btn = document.getElementById('btn-predict');

    if (!form || !btn) {
        return;
    }

    form.addEventListener('submit', function () {
        if (typeof form.reportValidity === 'function' && !form.reportValidity()) {
            return;
        }
        btn.classList.add('is-loading');
        btn.querySelector('.btn-label').textContent = 'Predicting…';
    });
});
