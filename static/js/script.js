document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('textInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const buttonContent = document.querySelector('.button-content');
    const loader = document.querySelector('.loader');

    textInput.addEventListener('input', () => {
        analyzeBtn.disabled = !textInput.value.trim();
    });

    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) return;

        // Show loading state
        analyzeBtn.disabled = true;
        buttonContent.classList.add('hidden');
        loader.classList.remove('hidden');

        try {
            const response = await fetch('http://127.0.0.1:8000/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                const errorText = await response.text(); // Get full error message
                throw new Error(`Server Error: ${response.status} - ${errorText}`);
            }

            const data = await response.json();

            if (!data.sentiment || data.confidence === undefined) {
                throw new Error("Invalid response format from server.");
            }

            // Redirect to results page with sentiment data in URL
            const queryParams = new URLSearchParams({
                sentiment: data.sentiment,
                confidence: (data.confidence * 100).toFixed(1)
            }).toString();
            window.location.href = `/result?${queryParams}`;

        } catch (error) {
            console.error('Error:', error.message);
            alert(`Error analyzing sentiment: ${error.message}`);
        } finally {
            analyzeBtn.disabled = false;
            buttonContent.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });
});
