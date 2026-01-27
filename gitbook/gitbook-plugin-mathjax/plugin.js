require(["gitbook"], function(gitbook) {
    function configureMathJax() {
        if (typeof MathJax === 'undefined' || !MathJax.Hub) {
            // Wait for MathJax to load
            setTimeout(configureMathJax, 200);
            return;
        }

        MathJax.Hub.Config({
            tex2jax: {
                inlineMath: [['$','$'], ['\\(','\\)']],
                processEscapes: true,
                ignoreClass: "tex2jax_ignore|dno"
            },
            showProcessingMessages: false,
            messageStyle: "none"
        });
        
        // Queue the typeset
        MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
    }

    gitbook.events.bind("page.change", function() {
        configureMathJax();
    });
    
    // Initial load
    configureMathJax();
});
