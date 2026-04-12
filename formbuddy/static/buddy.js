(function () {
    // Extract session ID from URL: /buddy/{session_id}
    const parts = window.location.pathname.split("/");
    const SESSION_ID = parts[parts.length - 1];

    const frame = document.getElementById("report-frame");
    const statusEl = document.getElementById("buddy-status");
    const previewEl = document.getElementById("buddy-preview");
    const sendBtn = document.getElementById("send-context");
    const fillBtn = document.getElementById("autofill-form");
    const toggleBtn = document.getElementById("buddy-toggle");
    const panel = document.getElementById("buddy-panel");
    const header = document.getElementById("buddy-header");

    let lastSuggestion = null;

    // --- Panel toggle ---
    header.addEventListener("click", function () {
        panel.classList.toggle("collapsed");
        toggleBtn.textContent = panel.classList.contains("collapsed") ? "▲" : "▼";
    });

    // --- Load session ---
    async function init() {
        try {
            const resp = await fetch(`/api/sessions/${SESSION_ID}`);
            if (!resp.ok) throw new Error("Session not found");
            const data = await resp.json();
            frame.src = data.proxy_url;
            statusEl.textContent = "Ready. Navigate the form, then tap Send Context.";
            sendBtn.disabled = false;
        } catch (e) {
            statusEl.textContent = "Error: " + e.message;
        }
    }

    // --- Extract visible fields from iframe ---
    function extractFields() {
        let doc;
        try {
            doc = frame.contentDocument || frame.contentWindow.document;
        } catch (e) {
            statusEl.textContent = "Cannot access form (cross-origin). Reload may help.";
            return null;
        }

        const fields = [];
        const inputs = doc.querySelectorAll(
            'input:not([type="hidden"]):not([type="submit"]):not([type="button"]):not([disabled]), ' +
            "textarea:not([disabled]), " +
            "select:not([disabled])"
        );

        inputs.forEach(function (el) {
            // Skip invisible elements
            if (el.offsetParent === null && el.type !== "radio" && el.type !== "checkbox") return;
            // Skip formbuddy injected scripts
            if (el.closest("[data-formbuddy]")) return;

            const fieldId = el.id || el.name || "";
            if (!fieldId) return;

            // Find label
            let label = "";
            if (el.id) {
                const labelEl = doc.querySelector('label[for="' + el.id + '"]');
                if (labelEl) label = labelEl.textContent.trim();
            }
            if (!label) {
                // Try closest label wrapper or preceding sibling
                const parentLabel = el.closest("label");
                if (parentLabel) label = parentLabel.textContent.trim();
            }
            if (!label) {
                // Try aria-label
                label = el.getAttribute("aria-label") || "";
            }

            let type = el.tagName.toLowerCase();
            if (type === "input") type = el.type || "text";

            const options = [];
            if (type === "select" || el.tagName === "SELECT") {
                type = "select";
                Array.from(el.options).forEach(function (opt) {
                    if (opt.value) options.push(opt.text || opt.value);
                });
            }
            if (type === "radio") {
                // Collect all options for this radio group
                const radios = doc.querySelectorAll('input[name="' + el.name + '"]');
                radios.forEach(function (r) {
                    const rl = doc.querySelector('label[for="' + r.id + '"]');
                    options.push(rl ? rl.textContent.trim() : r.value);
                });
            }

            // Avoid duplicate radio entries
            if (type === "radio" && fields.some(function (f) { return f.field_id === el.name; })) return;

            let selector = "";
            if (el.id) selector = "#" + el.id;
            else if (el.name) selector = '[name="' + el.name + '"]';

            fields.push({
                field_id: fieldId,
                selector: selector,
                label: label,
                type: type,
                required: el.required || false,
                placeholder: el.placeholder || "",
                options: options,
                current_value: el.value || "",
            });
        });

        return fields;
    }

    // --- Get page context from iframe ---
    function getPageContext() {
        try {
            const doc = frame.contentDocument || frame.contentWindow.document;
            return {
                title: doc.title || "",
                url: frame.contentWindow.location.href || "",
            };
        } catch (e) {
            return { title: "", url: "" };
        }
    }

    // --- Send Context ---
    sendBtn.addEventListener("click", async function () {
        sendBtn.disabled = true;
        fillBtn.disabled = true;
        statusEl.innerHTML = '<span class="spinner"></span>Extracting fields…';

        const fields = extractFields();
        if (!fields) {
            sendBtn.disabled = false;
            return;
        }

        if (fields.length === 0) {
            statusEl.textContent = "No form fields found on this page.";
            sendBtn.disabled = false;
            return;
        }

        statusEl.innerHTML =
            '<span class="spinner"></span>Sending ' + fields.length + " fields to LLM…";

        try {
            const resp = await fetch(`/api/sessions/${SESSION_ID}/suggest`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    page_context: getPageContext(),
                    fields: fields,
                }),
            });

            if (!resp.ok) {
                const err = await resp.text();
                throw new Error(err);
            }

            lastSuggestion = await resp.json();

            // Show result
            let msg = lastSuggestion.assistant_message || "Got suggestions.";
            if (lastSuggestion.fills && lastSuggestion.fills.length > 0) {
                msg += "\n\nSuggested fills:";
                lastSuggestion.fills.forEach(function (f) {
                    msg += "\n  • " + f.field_id + " = " + JSON.stringify(f.value);
                    if (f.reason) msg += " (" + f.reason + ")";
                });
            }
            if (lastSuggestion.needs_user_input && lastSuggestion.needs_user_input.length > 0) {
                msg += "\n\nNeeds your input: " + lastSuggestion.needs_user_input.join(", ");
            }

            statusEl.textContent = "✅ Suggestions ready. Review below, then tap Autofill.";
            previewEl.textContent = msg;
            previewEl.classList.add("visible");
            fillBtn.disabled = false;
        } catch (e) {
            statusEl.textContent = "❌ Error: " + e.message;
        } finally {
            sendBtn.disabled = false;
        }
    });

    // --- Autofill Form ---
    fillBtn.addEventListener("click", function () {
        if (!lastSuggestion || !lastSuggestion.fills) return;

        let doc;
        try {
            doc = frame.contentDocument || frame.contentWindow.document;
        } catch (e) {
            statusEl.textContent = "Cannot access form.";
            return;
        }

        let filled = 0;
        lastSuggestion.fills.forEach(function (fill) {
            // Find element by selector from the extracted fields
            let el = null;
            if (fill.field_id) {
                el = doc.getElementById(fill.field_id);
                if (!el) el = doc.querySelector('[name="' + fill.field_id + '"]');
            }
            if (!el) return;

            const tag = el.tagName.toLowerCase();

            if (tag === "select") {
                // Find matching option by text or value
                let matched = false;
                Array.from(el.options).forEach(function (opt) {
                    if (
                        opt.text.trim().toLowerCase() === String(fill.value).toLowerCase() ||
                        opt.value.toLowerCase() === String(fill.value).toLowerCase()
                    ) {
                        el.value = opt.value;
                        matched = true;
                    }
                });
                if (matched) filled++;
            } else if (el.type === "checkbox") {
                el.checked = fill.value === true || fill.value === "true" || fill.value === "yes";
                filled++;
            } else if (el.type === "radio") {
                const radios = doc.querySelectorAll('[name="' + el.name + '"]');
                radios.forEach(function (r) {
                    const rl = doc.querySelector('label[for="' + r.id + '"]');
                    const labelText = rl ? rl.textContent.trim().toLowerCase() : "";
                    if (
                        r.value.toLowerCase() === String(fill.value).toLowerCase() ||
                        labelText === String(fill.value).toLowerCase()
                    ) {
                        r.checked = true;
                        filled++;
                    }
                });
            } else {
                el.value = fill.value;
                filled++;
            }

            // Dispatch events so the form's JS picks up the changes
            el.dispatchEvent(new Event("input", { bubbles: true }));
            el.dispatchEvent(new Event("change", { bubbles: true }));
            el.dispatchEvent(new Event("blur", { bubbles: true }));
        });

        statusEl.textContent =
            "✅ Filled " + filled + " of " + lastSuggestion.fills.length + " fields. Review and continue.";
    });

    // --- Init ---
    init();
})();
