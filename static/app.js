const $ = (selector) =>
    document.querySelector(selector);


/* =========================================================
STATE
========================================================= */

const STORAGE_KEY =
    "BEHRAD_AI_TEST_CHATS_V1";


let chats =
    JSON.parse(
        localStorage.getItem(STORAGE_KEY)
        || "[]"
    );


let currentChatId = null;

let isBusy = false;

let chartCounter = 0;


/* =========================================================
MARKDOWN
========================================================= */

marked.setOptions({
    gfm: true,
    breaks: true
});


function renderMarkdown(text) {

    return DOMPurify.sanitize(
        marked.parse(text || "")
    );
}


function escapeHTML(value) {

    const div =
        document.createElement("div");

    div.textContent =
        String(value ?? "");

    return div.innerHTML;
}


function escapeAttribute(value) {

    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}


/* =========================================================
STORAGE
========================================================= */

function saveChats() {

    localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(chats)
    );
}


function getCurrentChat() {

    return chats.find(
        chat =>
            chat.id === currentChatId
    );
}


/* =========================================================
CHAT MANAGEMENT
========================================================= */

function createNewChat() {

    const chat = {

        id:
            crypto.randomUUID(),

        title:
            "چت جدید",

        messages:
            []
    };


    chats.unshift(chat);

    currentChatId =
        chat.id;

    saveChats();

    renderChatList();

    renderChat();

    closeSidebar();
}


function makeChatTitle(text) {

    return text
        .trim()
        .replace(/\s+/g, " ")
        .slice(0, 42)
        || "چت جدید";
}


function ensureChat() {

    if (
        !currentChatId
        || !getCurrentChat()
    ) {

        createNewChat();
    }
}


/* =========================================================
CHAT LIST
========================================================= */

function renderChatList() {

    const list =
        $("#chatList");

    list.innerHTML = "";


    chats.forEach(chat => {

        const button =
            document.createElement("button");


        button.className =
            "chat-item";


        if (
            chat.id === currentChatId
        ) {

            button.classList.add(
                "active"
            );
        }


        button.textContent =
            chat.title;


        button.title =
            chat.title;


        button.addEventListener(
            "click",
            () => {

                currentChatId =
                    chat.id;

                renderChatList();

                renderChat();

                closeSidebar();
            }
        );


        list.appendChild(button);

    });
}


/* =========================================================
CHAT RENDER
========================================================= */

function renderChat() {

    const chat =
        getCurrentChat();


    const container =
        $("#chat");


    container.innerHTML = "";


    if (
        !chat
        || chat.messages.length === 0
    ) {

        renderWelcome();

        return;
    }


    chat.messages.forEach(
        message => {

            if (
                message.role === "user"
            ) {

                appendUserMessage(
                    message.content
                );

                return;
            }


            if (
                message.role === "assistant"
            ) {

                const bubble =
                    appendAssistantMessage(
                        message.content || ""
                    );


                if (message.image) {

                    appendImage(
                        bubble,
                        message.image
                    );
                }


                if (
                    Array.isArray(
                        message.sources
                    )
                    && message.sources.length
                ) {

                    appendSources(
                        bubble,
                        message.sources
                    );
                }


                if (message.chart) {

                    appendChart(
                        bubble,
                        message.chart
                    );
                }

            }

        }
    );


    scrollToBottom();
}


/* =========================================================
WELCOME
========================================================= */

function renderWelcome() {

    const container =
        $("#chat");


    container.innerHTML = `

        <div class="welcome">

            <div class="welcome-icon">
                B
            </div>

            <h1>
                چه کاری می‌تونم برات انجام بدم؟
            </h1>

            <div class="suggestions">

                <button
                    data-prompt="جدیدترین قیمت RX 9070 را در وب پیدا کن و منابع را هم نشان بده."
                >
                    <span>🔎</span>
                    جستجوی وب
                </button>

                <button
                    data-prompt="یک گربه فضانورد روی ماه بساز."
                >
                    <span>🖼️</span>
                    ساخت تصویر
                </button>

                <button
                    data-prompt="برای اعداد 10، 20، 15، 30 یک نمودار بساز."
                >
                    <span>📊</span>
                    نمودار
                </button>

                <button
                    data-prompt="یک جدول مقایسه‌ای برای چند کارت گرافیک بساز."
                >
                    <span>📋</span>
                    جدول
                </button>

            </div>

        </div>
    `;


    bindSuggestionButtons();
}


/* =========================================================
MESSAGES
========================================================= */

function appendUserMessage(text) {

    const row =
        document.createElement("div");


    row.className =
        "message user";


    const bubble =
        document.createElement("div");


    bubble.className =
        "bubble";


    bubble.innerHTML =
        escapeHTML(text);


    row.appendChild(
        bubble
    );


    $("#chat").appendChild(
        row
    );

    scrollToBottom();
}


function appendAssistantMessage(
    text = ""
) {

    const row =
        document.createElement("div");


    row.className =
        "message assistant";


    const bubble =
        document.createElement("div");


    bubble.className =
        "bubble";


    bubble.innerHTML =
        renderMarkdown(text);


    row.appendChild(
        bubble
    );


    $("#chat").appendChild(
        row
    );


    scrollToBottom();


    return bubble;
}


/* =========================================================
ACTIVITY
========================================================= */

function addActivity(text) {

    const element =
        document.createElement("div");


    element.className =
        "activity";


    element.innerHTML = `

        <span class="activity-dot"></span>

        <span>
            ${escapeHTML(text)}
        </span>

    `;


    $("#chat").appendChild(
        element
    );


    scrollToBottom();


    return element;
}


/* =========================================================
IMAGE
========================================================= */

function appendImage(
    bubble,
    src
) {

    const image =
        document.createElement("img");


    image.className =
        "generated-image";


    image.src =
        src;


    image.alt =
        "تصویر تولیدشده توسط BEHRAD AI";


    image.loading =
        "lazy";


    bubble.appendChild(
        image
    );
}


/* =========================================================
SOURCES
========================================================= */

function appendSources(
    bubble,
    sources
) {

    if (!sources.length) {
        return;
    }


    const wrapper =
        document.createElement("div");


    wrapper.className =
        "sources";


    sources.forEach(
        source => {

            const link =
                document.createElement("a");


            link.className =
                "source";


            link.href =
                source.url;


            link.target =
                "_blank";


            link.rel =
                "noopener noreferrer";


            const favicon =
                escapeAttribute(
                    source.favicon
                    || ""
                );


            const title =
                escapeHTML(
                    source.title
                    || source.domain
                    || "منبع"
                );


            const domain =
                escapeHTML(
                    source.domain
                    || ""
                );


            link.innerHTML = `

                <img
                    src="${favicon}"
                    alt=""
                    loading="lazy"
                >

                <div class="source-text">

                    <b class="source-title">
                        ${title}
                    </b>

                    <small class="source-domain">
                        ${domain}
                    </small>

                </div>

            `;


            wrapper.appendChild(
                link
            );

        }
    );


    bubble.appendChild(
        wrapper
    );
}


/* =========================================================
CHART
========================================================= */

function appendChart(
    bubble,
    data
) {

    if (
        !Array.isArray(data.labels)
        || !Array.isArray(data.values)
        || data.labels.length !== data.values.length
    ) {

        return;
    }


    const wrapper =
        document.createElement("div");


    wrapper.className =
        "chart-wrapper";


    const canvas =
        document.createElement("canvas");


    chartCounter += 1;

    canvas.id =
        `behrad-chart-${chartCounter}`;


    wrapper.appendChild(
        canvas
    );


    bubble.appendChild(
        wrapper
    );


    new Chart(
        canvas,
        {

            type:
                data.type || "bar",

            data: {

                labels:
                    data.labels,

                datasets: [

                    {

                        label:
                            data.unit || "مقدار",

                        data:
                            data.values,

                        borderWidth:
                            2,

                        borderRadius:
                            7
                    }

                ]
            },

            options: {

                responsive: true,

                maintainAspectRatio: true,

                plugins: {

                    legend: {

                        display:
                            Boolean(
                                data.unit
                            )
                    },

                    title: {

                        display:
                            Boolean(
                                data.title
                            ),

                        text:
                            data.title || ""
                    }
                }
            }

        }
    );
}


/* =========================================================
SUGGESTIONS
========================================================= */

function bindSuggestionButtons() {

    document
        .querySelectorAll(
            "[data-prompt]"
        )
        .forEach(
            button => {

                button.onclick =
                    () => {

                        $("#messageInput")
                            .value =
                            button.dataset.prompt;

                        submitMessage();
                    };

            }
        );
}


/* =========================================================
SCROLL
========================================================= */

function scrollToBottom() {

    requestAnimationFrame(
        () => {

            const chat =
                $("#chat");

            chat.scrollTop =
                chat.scrollHeight;
        }
    );
}


/* =========================================================
TEXTAREA
========================================================= */

function resizeTextarea() {

    const input =
        $("#messageInput");


    input.style.height =
        "auto";


    input.style.height =
        Math.min(
            input.scrollHeight,
            180
        ) + "px";
}


/* =========================================================
SEND
========================================================= */

async function submitMessage() {

    if (isBusy) {
        return;
    }


    const input =
        $("#messageInput");


    const text =
        input.value.trim();


    if (!text) {
        return;
    }


    ensureChat();


    const chat =
        getCurrentChat();


    if (
        chat.messages.length === 0
        || chat.title === "چت جدید"
    ) {

        chat.title =
            makeChatTitle(text);
    }


    chat.messages.push({

        role:
            "user",

        content:
            text

    });


    saveChats();

    renderChatList();


    if (
        document.querySelector(
            ".welcome"
        )
    ) {

        renderChat();

    } else {

        appendUserMessage(
            text
        );
    }


    input.value = "";

    resizeTextarea();


    isBusy = true;

    $("#sendButton").disabled =
        true;


    const assistantBubble =
        appendAssistantMessage(
            ""
        );


    let answer = "";

    let generatedImage =
        null;

    let sources = [];

    let chart = null;

    const activities = [];


    try {

        const response =
            await fetch(
                "/api/chat",
                {

                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            messages:
                                chat.messages
                        })
                }
            );


        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(
                errorText
            );
        }


        if (!response.body) {

            throw new Error(
                "Streaming response is unavailable."
            );
        }


        const reader =
            response.body.getReader();


        const decoder =
            new TextDecoder();


        let buffer = "";


        while (true) {

            const {
                value,
                done
            } =
                await reader.read();


            if (done) {
                break;
            }


            buffer +=
                decoder.decode(
                    value,
                    {
                        stream: true
                    }
                );


            const events =
                buffer.split("\n\n");


            buffer =
                events.pop()
                || "";


            for (
                const rawEvent
                of events
            ) {

                const match =
                    rawEvent.match(
                        /^event:\s*(.+)\ndata:\s*(.+)$/s
                    );


                if (!match) {
                    continue;
                }


                const eventName =
                    match[1];


                let data;


                try {

                    data =
                        JSON.parse(
                            match[2]
                        );

                } catch {

                    continue;
                }


                /* STATUS */

                if (
                    eventName === "status"
                ) {

                    const activity =
                        addActivity(
                            data.text
                        );


                    activities.push(
                        activity
                    );
                }


                /* TEXT */

                if (
                    eventName === "delta"
                ) {

                    answer +=
                        data.text || "";


                    assistantBubble.innerHTML =
                        renderMarkdown(
                            answer
                        );


                    scrollToBottom();
                }


                /* SOURCES */

                if (
                    eventName === "sources"
                ) {

                    const items =
                        data.items
                        || [];


                    sources.push(
                        ...items
                    );


                    appendSources(
                        assistantBubble,
                        items
                    );


                    scrollToBottom();
                }


                /* IMAGE */

                if (
                    eventName === "image"
                ) {

                    generatedImage =
                        data.image;


                    appendImage(
                        assistantBubble,
                        generatedImage
                    );


                    scrollToBottom();
                }


                /* CHART */

                if (
                    eventName === "chart"
                ) {

                    chart =
                        data;


                    appendChart(
                        assistantBubble,
                        chart
                    );


                    scrollToBottom();
                }


                /* ERROR */

                if (
                    eventName === "error"
                ) {

                    throw new Error(
                        data.message
                        || "خطای ناشناخته"
                    );
                }

            }

        }


        /* REMOVE ACTIVITY */

        activities.forEach(
            activity =>
                activity.remove()
        );


        /* SAVE ASSISTANT */

        chat.messages.push({

            role:
                "assistant",

            content:
                answer,

            image:
                generatedImage,

            sources:
                sources,

            chart:
                chart
        });


        saveChats();


    } catch (error) {

        activities.forEach(
            activity =>
                activity.remove()
        );


        assistantBubble.innerHTML = `

            <b>
                خطا:
            </b>

            ${escapeHTML(
                error.message
            )}

        `;

    } finally {

        isBusy =
            false;

        $("#sendButton").disabled =
            false;

        scrollToBottom();
    }
}


/* =========================================================
SIDEBAR
========================================================= */

const sidebar =
    $("#sidebar");


const overlay =
    $("#mobileOverlay");


function openSidebar() {

    sidebar.classList.add(
        "open"
    );

    overlay.classList.add(
        "show"
    );
}


function closeSidebar() {

    sidebar.classList.remove(
        "open"
    );

    overlay.classList.remove(
        "show"
    );
}


$("#menuButton")
    .addEventListener(
        "click",
        openSidebar
    );


$("#closeSidebar")
    .addEventListener(
        "click",
        closeSidebar
    );


overlay.addEventListener(
    "click",
    closeSidebar
);


/* =========================================================
BUTTONS
========================================================= */

$("#newChatButton")
    .addEventListener(
        "click",
        createNewChat
    );


$("#topNewChat")
    .addEventListener(
        "click",
        createNewChat
    );


$("#composer")
    .addEventListener(
        "submit",
        event => {

            event.preventDefault();

            submitMessage();
        }
    );


$("#messageInput")
    .addEventListener(
        "input",
        resizeTextarea
    );


$("#messageInput")
    .addEventListener(
        "keydown",
        event => {

            if (
                event.key === "Enter"
                && !event.shiftKey
            ) {

                event.preventDefault();

                submitMessage();
            }

        }
    );


/* =========================================================
INIT
========================================================= */

if (!Array.isArray(chats)) {

    chats = [];
}


if (chats.length === 0) {

    createNewChat();

} else {

    currentChatId =
        chats[0].id;

    renderChatList();

    renderChat();
}


resizeTextarea();
