#!/bin/bash

# Patch index.js
TARGET_FILE="node_modules/gitbook-plugin-mathjax/index.js"
if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: $TARGET_FILE not found. Please run 'npm install' or 'gitbook install' first."
    exit 1
fi

cat > "$TARGET_FILE" << 'INNEREOF'
var Q = require('q');
var fs = require('fs');
var path = require('path');
var crc = require('crc');
var exec = require('child_process').exec;

// Lazily require mathjax-node.
//
// Reason: For website builds (our primary use), this plugin can emit
// <script type="math/tex"> blocks and rely on client-side MathJax.
// Requiring `mathjax-node` pulls in a modern `mathjax` dependency that uses
// optional chaining (?.), which is not supported by the Node runtime
// commonly used with GitBook legacy.
var mjAPI = null;

var started = false;
var countMath = 0;
var cache = {};

/**
    Prepare MathJaX
*/
function prepareMathJax() {
    if (started) {
        return;
    }

    if (!mjAPI) {
        mjAPI = require('mathjax-node/lib/mj-single.js');
    }

    mjAPI.config({
        MathJax: {
            SVG: {
                font: 'TeX'
            }
        }
    });
    mjAPI.start();

    started = true;
}

/**
    Convert a tex formula into a SVG text

    @param {String} tex
    @param {Object} options
    @return {Promise<String>}
*/
function convertTexToSvg(tex, options) {
    var d = Q.defer();
    options = options || {};

    prepareMathJax();

    mjAPI.typeset({
        math:           tex,
        format:         (options.inline ? 'inline-TeX' : 'TeX'),
        svg:            true,
        speakText:      true,
        speakRuleset:   'mathspeak',
        speakStyle:     'default',
        ex:             6,
        width:          100,
        linebreaks:     true
    }, function (data) {
        if (data.errors) {
            return d.reject(new Error(data.errors));
        }

        d.resolve(options.write? null : data.svg);
    });

    return d.promise;
}

/**
    Process a math block

    @param {Block} blk
    @return {Promise<Block>}
*/
function processBlock(blk) {
    var book = this;
    var tex = blk.body;
    var isInline = !(tex[0] == "\n");

    // For website return as script
    var config = book.config.get('pluginsConfig.mathjax', {});

    if ((book.output.name == "website" || book.output.name == "json")
        && !config.forceSVG) {
        return '<script type="math/tex'+(isInline? "": "; mode=display")+'">'+blk.body+'</script>';
    }

    // Check if not already cached
    var hashTex = crc.crc32(tex).toString(16);

    // Return
    var imgFilename = '_mathjax_' + hashTex + '.svg';
    var img = '<img src="/' + imgFilename + '" />';

    // Center math block
    if (!isInline) {
        img = '<div style="text-align:center;margin: 1em 0em;width: 100%;">' + img + '</div>';
    }

    return {
        body: img,
        post: function() {
            if (cache[hashTex]) {
                return;
            }

            cache[hashTex] = true;
            countMath = countMath + 1;

            return convertTexToSvg(tex, { inline: isInline })
            .then(function(svg) {
                return book.output.writeFile(imgFilename, svg);
            });
        }
    };
}

/**
    Return assets for website

    @return {Object}
*/
function getWebsiteAssets() {
    // The historic cdn.mathjax.org endpoint is deprecated and no longer
    // reliably serves MathJax. Use a stable CDN by default.
    var cdnUrl = this.config.get(
        'pluginsConfig.mathjax.cdnUrl',
        'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
    );

    return {
        assets: "./book",
        js: [
            cdnUrl,
            'plugin.js'
        ]
    };
}

module.exports = {
    website: getWebsiteAssets,
    blocks: {
        math: {
            shortcuts: {
                parsers: ["markdown", "asciidoc"],
                start: "$$",
                end: "$$"
            },
            process: processBlock
        }
    }
};
INNEREOF
echo "Patched index.js"

# Patch plugin.js
PLUGIN_FILE="node_modules/gitbook-plugin-mathjax/book/plugin.js"
if [ ! -f "$PLUGIN_FILE" ]; then
    echo "Error: $PLUGIN_FILE not found."
    exit 1
fi

cat > "$PLUGIN_FILE" << 'INNEREOF'
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
INNEREOF
echo "Patched plugin.js"

echo "Successfully patched gitbook-plugin-mathjax."
