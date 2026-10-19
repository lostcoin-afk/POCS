const $ = (id) => document.getElementById(id);
let latestData = null;
let debounceTimer = null;
let toastTimer = null;
const defaults = {
  canny_low:45, canny_high:130, blur_kernel:5, morph_kernel:5, roi_top_pct:0,
  min_area_ratio:0.015, circle_min_radius:6, circle_max_radius:140, circle_min_distance:58, circle_param2:34, circle_min_contrast:16,
  line_hough_threshold:55, line_min_length:75, line_max_gap:20, axis_tolerance_deg:10,
  line_merge_px:13, max_axis_lines:8, min_dimension_px:18
};
function toast(message, error=false){
  const node=$('toast'); node.textContent=message; node.style.borderColor=error?'#a74d5a':'#3e566a'; node.classList.add('show');
  clearTimeout(toastTimer); toastTimer=setTimeout(()=>node.classList.remove('show'),3200);
}
async function api(url, options={}){
  const response=await fetch(url,{headers:{'Content-Type':'application/json',...(options.headers||{})},...options});
  const data=await response.json().catch(()=>({detail:response.statusText}));
  if(!response.ok) throw new Error(data.detail||data.message||`Request failed (${response.status})`);
  return data;
}
function showSourceFields(){
  const type=$('source-type').value;
  $('video-fields').classList.toggle('hidden',type!=='video');
  $('arena-fields').classList.toggle('hidden',type!=='arena');
  $('webcam-fields').classList.toggle('hidden',type!=='webcam');
}
function currentSource(){return {
  source_type:$('source-type').value,
  video_path:$('video-path').value.trim(),
  webcam_index:Number($('webcam-index').value||0),
  arena_serial:$('arena-serial').value.trim(),
  loop_video:$('loop-video').checked
};}
function getProcessingSettings(){return {
  canny_low:Number($('canny-low').value), canny_high:Number($('canny-high').value),
  blur_kernel:Number($('blur-kernel').value), morph_kernel:Number($('morph-kernel').value),
  min_area_ratio:Number($('min-area').value), roi_top_pct:Number($('roi-top-pct').value),
  circle_min_radius:Number($('circle-min-radius').value), circle_max_radius:Number($('circle-max-radius').value),
  circle_min_distance:Number($('circle-min-distance').value), circle_param2:Number($('circle-param2').value),
  circle_min_contrast:Number($('circle-contrast').value),
  line_hough_threshold:Number($('line-hough-threshold').value), line_min_length:Number($('line-min-length').value),
  line_max_gap:Number($('line-max-gap').value), axis_tolerance_deg:Number($('axis-tolerance-deg').value),
  line_merge_px:Number($('line-merge-px').value), max_axis_lines:Number($('max-axis-lines').value),
  min_dimension_px:Number($('min-dimension-px').value)
};}
function updateSettingOutputs(){
  const map={'roi-top-pct':'roi-top-pct-out','canny-low':'canny-low-out','canny-high':'canny-high-out','blur-kernel':'blur-kernel-out','morph-kernel':'morph-kernel-out','min-area':'min-area-out','circle-min-radius':'circle-min-radius-out','circle-max-radius':'circle-max-radius-out','circle-min-distance':'circle-min-distance-out','circle-param2':'circle-param2-out','circle-contrast':'circle-contrast-out','line-hough-threshold':'line-hough-threshold-out','line-min-length':'line-min-length-out','line-max-gap':'line-max-gap-out','axis-tolerance-deg':'axis-tolerance-deg-out','line-merge-px':'line-merge-px-out','max-axis-lines':'max-axis-lines-out','min-dimension-px':'min-dimension-px-out'};
  for(const [input,out] of Object.entries(map)) $(out).textContent=$(input).value;
}
async function pushSettings(){
  try{await api('/api/settings',{method:'POST',body:JSON.stringify(getProcessingSettings())});}
  catch(e){toast(e.message,true);}
}
function queueSettings(){updateSettingOutputs();clearTimeout(debounceTimer);debounceTimer=setTimeout(pushSettings,180);}
function displayLength(value, label, px){
  if(value===null||value===undefined) return '—';
  return label || `${Number(value).toFixed(1)} px`;
}
function renderCircles(circles){
  const body=$('circles-body');
  if(!circles||circles.length===0){body.innerHTML='<tr><td colspan="6" class="empty-cell">No circles passed the current detection filters.</td></tr>';return;}
  body.innerHTML=circles.map(c=>`<tr><td class="circle-id">${c.id}</td><td>${Number(c.cx).toFixed(1)}, ${Number(c.cy).toFixed(1)}</td><td>${Number(c.radius_px).toFixed(1)}</td><td>${Number(c.diameter_px).toFixed(1)} px</td><td>${c.diameter_label}</td><td>${Number(c.contrast).toFixed(1)}</td></tr>`).join('');
}
function renderLinearMeasurements(items){
  const body=$('linear-body');
  if(!items||items.length===0){body.innerHTML='<tr><td colspan="6" class="empty-cell">No line-to-line measurements found. Reduce line threshold or segment length.</td></tr>';return;}
  const sorted=[...items].sort((a,b)=>{
    if(a.overall&&!b.overall)return -1;if(!a.overall&&b.overall)return 1;
    if(a.axis!==b.axis)return a.axis.localeCompare(b.axis);
    return a.id.localeCompare(b.id);
  });
  body.innerHTML=sorted.map(m=>`<tr class="${m.overall?'overall-row':''}"><td class="circle-id">${m.id}</td><td>${m.axis==='X'?'Horizontal':'Vertical'}</td><td>${m.line_a} ↔ ${m.line_b}</td><td>${Number(m.distance_px).toFixed(1)} px</td><td>${m.distance_label}</td><td>${m.overall?'Overall span':'Adjacent guides'}</td></tr>`).join('');
}
function renderStatus(payload){
  const s=payload.status||{},d=payload.data||{},p=payload.processing||{};latestData=d;
  $('connection-dot').classList.toggle('active',Boolean(s.running));
  $('connection-label').textContent=s.running?'STREAMING':(s.error?'SOURCE ERROR':'DISCONNECTED');
  $('source-name').textContent=s.source_label||'No source selected';
  $('fps-label').textContent=`${Number(s.fps||0).toFixed(1)} FPS`;
  $('source-message').textContent=s.error||s.message||'—';$('source-message').classList.toggle('error',Boolean(s.error));
  $('feed-placeholder').classList.toggle('show',!s.running && !d.silhouette?.found);
  $('feed-placeholder').textContent=s.error||'Select a video or camera source, then click Start measurement.';
  $('length-value').textContent=displayLength(d.length,d.length_label,d.length_px);
  $('width-value').textContent=displayLength(d.width,d.width_label,d.width_px);
  $('length-sub').textContent=d.length_px!=null?`${Number(d.length_px).toFixed(1)} px · horizontal silhouette span`:'Waiting for silhouette';
  $('width-sub').textContent=d.width_px!=null?`${Number(d.width_px).toFixed(1)} px · vertical silhouette span`:'Waiting for silhouette';
  $('circle-count').textContent=(d.circles||[]).length;
  $('processing-ms').textContent=d.processing_ms==null?'—':`${Number(d.processing_ms).toFixed(1)} ms`;
  $('frame-size').textContent=`Frame ${d.image_width||'—'} × ${d.image_height||'—'}`;
  $('silhouette-label').textContent=d.silhouette?.found?`Silhouette: ${(100*(d.silhouette.area_ratio||0)).toFixed(1)}% frame`:'Silhouette: not detected';
  if(d.roi){$('roi-state').textContent=`Active region: y=${d.roi.top_y}px (${Number(d.roi.top_pct).toFixed(1)}%), full width to bottom.`;}
  if(p.roi_top_pct!==undefined && document.activeElement!==$('roi-top-pct')){$('roi-top-pct').value=String(Math.round(Number(p.roi_top_pct)));$('roi-top-pct-out').textContent=`${$('roi-top-pct').value}%`;}
  $('diag-resolution').textContent=`${d.image_width||'—'} × ${d.image_height||'—'}`;
  $('diag-area').textContent=d.silhouette?.found?`${(100*d.silhouette.area_ratio).toFixed(2)}%`:'—';
  $('diag-method').textContent=d.silhouette?.method||'—';$('diag-edges').textContent=Number(d.edge_pixels||0).toLocaleString();
  $('diag-h-guides').textContent=String((d.horizontal_guides||[]).length);$('diag-v-guides').textContent=String((d.vertical_guides||[]).length);
  $('diag-linear-count').textContent=String((d.linear_measurements||[]).length);
  $('diag-fps').textContent=`${Number(s.fps||0).toFixed(1)}`;$('diag-time').textContent=d.processing_ms==null?'—':`${Number(d.processing_ms).toFixed(1)} ms`;
  if(d.calibration_resolution_matches===false){$('calibration-state').textContent='Calibration inactive: source resolution differs from calibration';$('calibration-state').style.color='#ff9b9b';}
  else if(d.calibration_mode==='planar homography'){$('calibration-state').textContent=`Planar homography active · nominal ${Number(p.mm_per_px||0).toFixed(6)} mm/px`;$('calibration-state').style.color='#8ce5b0';}
  else if(p.mm_per_px && d.calibration_mode!=='uncalibrated'){$('calibration-state').textContent=`Scalar calibration · ${Number(p.mm_per_px).toFixed(6)} mm/px`;$('calibration-state').style.color='#f0d28c';}
  else{$('calibration-state').textContent='Uncalibrated · measurements in pixels';$('calibration-state').style.color='#f0d28c';}
  renderCircles(d.circles||[]);renderLinearMeasurements(d.linear_measurements||[]);
}
async function pollStatus(){try{const status=await api('/api/status');renderStatus(status);}catch(e){$('source-message').textContent=e.message;}}
async function startSource(){
  try{
    await pushSettings();
    const result=await api('/api/start',{method:'POST',body:JSON.stringify(currentSource())});
    $('video-feed').src='/video_feed?t='+Date.now();
    document.querySelectorAll('.preview-image img').forEach((img,i)=>img.src=(i===0?'/mask_feed':'/edge_feed')+'?t='+Date.now());
    $('source-message').classList.remove('error');$('source-message').textContent=result.message;toast('Measurement source started.');
  }catch(e){toast(e.message,true);$('source-message').textContent=e.message;$('source-message').classList.add('error');}
}
async function stopSource(){try{const result=await api('/api/stop',{method:'POST',body:'{}'});$('source-message').textContent=result.message;toast('Capture stopped.');}catch(e){toast(e.message,true);}}
async function calibrate(){
  const known_length_px=Number($('known-px').value),known_length_mm=Number($('known-mm').value);
  try{const r=await api('/api/calibration',{method:'POST',body:JSON.stringify({known_length_px,known_length_mm})});toast(`${r.message} Scale: ${r.mm_per_px.toFixed(6)} mm/px`);await pollStatus();}
  catch(e){toast(e.message,true);}
}
async function clearCalibration(){try{await api('/api/calibration',{method:'DELETE'});toast('Calibration cleared; dimensions are in pixels.');await pollStatus();}catch(e){toast(e.message,true);}}
async function uploadVideo(file){
  if(!file)return;const form=new FormData();form.append('file',file);$('upload-status').textContent='Uploading…';
  try{const response=await fetch('/api/video/upload',{method:'POST',body:form});const r=await response.json();if(!response.ok)throw new Error(r.detail||'Upload failed');$('video-path').value=r.video_path;$('upload-status').textContent='Uploaded';toast('Video uploaded. Click Start measurement.');}
  catch(e){$('upload-status').textContent='';toast(e.message,true);}
}
function resetControls(){
  for(const [id,value] of Object.entries(defaults)){
    const map={min_area_ratio:'min-area',circle_min_contrast:'circle-contrast'};
    const node=$(map[id]||id.replaceAll('_','-'));
    if(node)node.value=value;
  }
  updateSettingOutputs();pushSettings();toast('Processing settings reset.');
}
$('source-type').addEventListener('change',showSourceFields);
$('start-btn').addEventListener('click',startSource);$('stop-btn').addEventListener('click',stopSource);
$('reset-btn').addEventListener('click',()=>{ $('video-feed').src='/video_feed?t='+Date.now();toast('Feed view refreshed.');});
$('apply-calibration').addEventListener('click',calibrate);$('clear-calibration').addEventListener('click',clearCalibration);
$('video-upload').addEventListener('change',e=>uploadVideo(e.target.files[0]));$('settings-reset').addEventListener('click',resetControls);
for(const id of ['roi-top-pct','canny-low','canny-high','blur-kernel','morph-kernel','min-area','circle-min-radius','circle-max-radius','circle-min-distance','circle-param2','circle-contrast','line-hough-threshold','line-min-length','line-max-gap','axis-tolerance-deg','line-merge-px','max-axis-lines','min-dimension-px'])$(id).addEventListener('input',queueSettings);
const feed=$('video-feed');
const feedStage=$('feed-stage');
let draggingRoi=false;
function setRoiFromPointer(event){
  const rect=feed.getBoundingClientRect();
  if(!rect.height)return;
  const pct=Math.max(0,Math.min(80,((event.clientY-rect.top)/rect.height)*100));
  $('roi-top-pct').value=String(Math.round(pct));
  updateSettingOutputs();
  $('roi-state').textContent=`ROI top cutoff: ${$('roi-top-pct').value}% of frame height.`;
  clearTimeout(debounceTimer);debounceTimer=setTimeout(pushSettings,120);
}
$('roi-top-pct').addEventListener('input',()=>{$('roi-state').textContent=`ROI top cutoff: ${$('roi-top-pct').value}% of frame height.`;});
$('roi-drag-enable').addEventListener('change',()=>{
  feedStage.classList.toggle('roi-drag-mode',$('roi-drag-enable').checked);
  toast($('roi-drag-enable').checked?'Drag vertically on the image to move the ROI top boundary.':'ROI drag adjustment disabled.');
});
feed.addEventListener('pointerdown',event=>{if(!$('roi-drag-enable').checked)return;event.preventDefault();draggingRoi=true;feed.setPointerCapture?.(event.pointerId);setRoiFromPointer(event);});
feed.addEventListener('pointermove',event=>{if(draggingRoi)setRoiFromPointer(event);});
feed.addEventListener('pointerup',event=>{draggingRoi=false;try{feed.releasePointerCapture?.(event.pointerId);}catch(_){}});
feed.addEventListener('pointercancel',()=>{draggingRoi=false;});
showSourceFields();updateSettingOutputs();pollStatus();setInterval(pollStatus,850);
