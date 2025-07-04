// Canvas drawing adapted from https://stackoverflow.com/questions/22891827/how-do-i-hand-draw-on-canvas-with-javascript

let canvas = document.getElementById('sheet')
let g = canvas.getContext("2d");

g.fillStyle = "black";
g.fillRect(0, 0, canvas.width, canvas.height);
g.strokeStyle = "white";
g.lineJoin = "round";
g.lineWidth = 20;
g.filter = "blur(1px) grayscale(1)";

const relPos = pt => [pt.pageX - canvas.offsetLeft, pt.pageY - canvas.offsetTop];
const drawStart = pt => { with(g) { beginPath(); moveTo.apply(g, pt); stroke(); }};
const drawMove = pt => { with(g) { lineTo.apply(g, pt); stroke(); }};

const pointerDown = e => drawStart(relPos(e.touches ? e.touches[0] : e));
const pointerMove = e => drawMove(relPos(e.touches ? e.touches[0] : e));

const draw = (method, move, stop) => e => {
    if(method=="add") pointerDown(e);
    canvas[method+"EventListener"](move, pointerMove);
    canvas[method+"EventListener"](stop, g.closePath);
};

canvas.addEventListener("mousedown", draw("add","mousemove","mouseup"));
canvas.addEventListener("touchstart", draw("add","touchmove","touchend"));
canvas.addEventListener("mouseup", draw("remove","mousemove","mouseup"));
canvas.addEventListener("touchend", draw("remove","touchmove","touchend"));

/** Scales the current canvas contents to 28x28 and sends the data to the server. */
async function predict(canvas, model, history) {
    // Create scaled-down version.
    let bitmap = await createImageBitmap(canvas, {
        resizeWidth: 28,
        resizeHeight: 28,
        resizeQuality: 'medium',
    });
    let sent_image = document.createElement("canvas");
    sent_image.width = 28;
    sent_image.height = 28;

    // Create canvas for history.
    let ctx = sent_image.getContext("bitmaprenderer");
    ctx.transferFromImageBitmap(bitmap);
    bitmap.close();
    
    // Make request to prediction REST endpoint.
    let url = sent_image.toDataURL();
    let postOptions = {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            label: model,
            data: url,
        }),
    };
    let response = await fetch('/predict/mnist', postOptions);

    if (response.ok) {
        let json = await response.json();
        let prediction = json.prediction;
        let full_prediction = json.full_prediction;

        // Add a history entry, discarding the oldest if necessary.
        let tbody = history.getElementsByTagName("tbody")[0]
        if (tbody.childNodes.length > 9) {
            tbody.removeChild(tbody.firstChild);
        }
        let row = document.createElement("tr");
        let model = document.createElement("td");
        model.innerText = json.label
        row.appendChild(model)
        im = document.createElement("td");
        im.addEventListener("click", async e => {
            let bitmap = await createImageBitmap(sent_image, {
                resizeWidth: 280,
                resizeHeight: 280,
                resizeQuality: 'medium',
            });
            let ctx = canvas.getContext("2d");
            ctx.drawImage(bitmap, 0, 0);
            bitmap.close();
            ctx.close();
        });
        res = document.createElement("td");
        res.innerText = prediction
        row.appendChild(im);
        im.appendChild(sent_image);
        row.appendChild(res);

        // Add the full prediction values and shade the background
        // according to the prediction value.
        for (let i of full_prediction) {
            full = document.createElement("td");
            full.innerText = i;
            b = Math.max(100, 255 - i * 255);
            full.style.backgroundColor = `rgb(${b},${b},${b})`;
            row.appendChild(full);
        }
        tbody.appendChild(row);
    } else {
        console.log("Problem with server: ", response);
    }
}

async function fill_models(select) {
    models = await (await fetch('/list/mnist')).json()
    models.forEach(model => {
        let option = document.createElement('option');
        option.innerText = model;
        option.setAttribute('value', model);
        select.appendChild(option);
    });
    select.value = models[0];
}

let select = document.getElementById('models');
fill_models(select)

let predict_button = document.getElementById('predict');
let history = document.getElementById('history');
predict_button.addEventListener("click", e => {
    predict(canvas, select.value, history);
    g.fillRect(0, 0, canvas.width, canvas.height);
});
let clear_button = document.getElementById('clear');
clear_button.addEventListener("click", e => {
    g.fillRect(0, 0, canvas.width, canvas.height);
});
