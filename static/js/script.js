Webcam.set({
  width: 400,
  height: 350,
  image_format: 'jpeg',
  jpeg_quality: 90
})

Webcam.attach("#camera")

function take_snapshot(){
 Webcam.snap(function(data_uri){
     /*document.getElementById('results').innerHTML = '<img src="'+data_uri+'"/>';*/
      var raw_image_data = data_uri.replace(/^data\:image\/\w+\;base64\,/, '');
      document.getElementById('mydata').value = raw_image_data;
      document.getElementById('myform').submit();

 })
     
}

function toggleview(){
  document.getElementById("permissiondiv").style.display="none";
  document.getElementById("cameradiv").style.display="flex";

}

//Drag and Drop Js

const dropArea = document.querySelector(".drag-area"),
// let dropArea = document.querySelector(".drag-area"),
dragText = dropArea.querySelector("header"),
button = dropArea.querySelector("button"),
input = dropArea.querySelector("input");

// declaring a global variable
let file; 
button.onclick = ()=>{
  input.click(); //onclick
}
input.addEventListener("change", function(){  //change of event
  file = this.files[0]; 
  dropArea.classList.add("active");
  showFile(); 
});

dropArea.addEventListener("dragenter", (event)=>{
  event.preventDefault();
});

// changing text when image dropover dragarea
dropArea.addEventListener("dragover", (event)=>{
  event.preventDefault();
  var dropEffect = updateDropEffect();  
  event.dataTransfer.effectAllowed = 'move';
  event.dataTransfer.dropEffect = "copymove";
  dropArea.classList.add("active");
  dragText.textContent = "Release to Upload File";
});

//If file not released over droparea left over it
dropArea.addEventListener("dragleave", ()=>{
  dropArea.classList.remove("active");
  dragText.textContent = "Drag & Drop to Upload File";
});

//If user drop File on droparea
dropArea.addEventListener("drop", (event)=>{
  event.preventDefault(); 
  console.log(event);
  //for handling multiple files
  console.log(event.dataTransfer.files[0]);
  file = event.dataTransfer.files[0];
  showFile(); 
});
function showFile(){
  let fileType = file.type; //getting selected file type
  let validExtensions = ["image/jpeg", "image/jpg", "image/png"]; 
  if(validExtensions.includes(fileType)){ //if user selected file is an image file
    console.log("hi",file)
    document.getElementById("form").submit();
  }else{
    alert("This is not an Image File!");
    dropArea.classList.remove("active");
    dragText.textContent = "Drag & Drop to Upload File";
  }
}