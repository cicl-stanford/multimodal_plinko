/* task.js
 * 
 * This file holds the main experiment code.
 * 
 * Requires:
 *   config.js
 *   psiturk.js
 *   utils.js
 */

// app.config['TEMPLATES_AUTO_RELOAD'] = True

// Create and initialize the experiment configuration object
var $c = new Config(condition, counterbalance);

// Initalize psiturk object
var psiTurk = new PsiTurk(uniqueId, adServerLoc);

// Preload the HTML template pages that we need for the experiment
psiTurk.preloadPages($c.pages);

// Objects to keep track of the current phase and state
var CURRENTVIEW;
var STATE;

/*************************™
 * INSTRUCTIONS         
 *************************/

 var Instructions = function() {
	
	$(".slide").hide();
	var slide = $("#instructions-training-0");
	slide.fadeIn($c.fade);
	// CURRENTVIEW = new TestPhase();
	// CURRENTVIEW = new Practice();
	
	slide.find('.next').click(function () {
		CURRENTVIEW = new Comprehension();
	});
};


/*****************
 *  COMPREHENSION CHECK QUESTIONS*
 *****************/

var Comprehension = function() {

	var that = this;

	// Show the slide
	$(".slide").hide();
	$("#comprehension_check").fadeIn($c.fade);

	//disable button initially
	$('#comprehension').prop('disabled', true);

	//checks whether all questions were answered
	$('.demoQ').change(function() {
		if ($('input[name=holes]:checked').length > 0 &&
			$('input[name=task]:checked').length > 0) {
			$('#comprehension').prop('disabled', false)
		} else {
			$('#comprehension').prop('disabled', true)
		}
	});

	$('#comprehension').click(function() {
		var q1 = $('input[name=holes]:checked').val();
		var q2 = $('input[name=task]:checked').val();

		// correct answers
		answers = ["3", "predict"]

		if (q1 == answers[0] && q2 == answers[1]) {
			CURRENTVIEW = new Practice();
		} else {
			$('input[name=holes]').prop('checked', false);
			$('input[name=task]').prop('checked', false);
			CURRENTVIEW = new ComprehensionCheckFail();
		}
	});
}

/*****************
 *  COMPREHENSION FAIL SCREEN*
 *****************/

var ComprehensionCheckFail = function() {
	// Show the slide
	$(".slide").hide();
	$("#comprehension_check_fail").fadeIn($c.fade);
	$('#start').unbind();
	$('#comprehension').unbind();

	$('#comprehension_fail').click(function() {
		CURRENTVIEW = new Instructions();
		$('#comprehension_fail').unbind();
	});
}


/*************************
 * PRACTICE
 *************************/

var Practice = function() {

	$(".slide").hide();
	$("#practice").fadeIn($c.fade);
	var that = this
	var counter = 0;
	var trial_counter = 0;
	var max_trials = 4;
	video_name = $c.videos[trial_counter].video

	this.load_video = function(video_name) {
		$("#video_mp4").attr("src", '/static/videos/mp4/' + video_name + '.mp4');
		$("#video_webm").attr("src", '/static/videos/webm/' + video_name + '.webm');
		$(".stim_video").load()
	}

	that.load_video(video_name);
	// disable button until end of video
	$(this).ready(function() {
		$('.stim_video').on('ended', function() {
			$("#play").prop('disabled', false);
		});
	});

	$("#play.next").click(function() {
		counter++;
		if (counter < 3) {
			$("#play").prop('disabled', true);
			// that.load_video(video_name);
			$('.stim_video').trigger('play');
		}
		if (counter == 1) {
			$('#play').html('Watch again');
		}
		if (counter == 2) {
			if (trial_counter <= (max_trials-2)) {
				console.log("trial_counter", trial_counter);
				// console.log("trial_counter", trial_counter);
				$('#play').html('See next video');
			} else {
				$('#play').html('Go to main experiment phase');
			}
		}
		if (counter == 3) {
			$('#play').html('Watch video');
			trial_counter++
			if (trial_counter == max_trials) {
				CURRENTVIEW = new TestPhase();
			}
			else{
			video_name = $c.videos[trial_counter].video
			that.load_video(video_name);
			counter = 0;
		}
		}
	});
}

/*************************
 * TRIAL
 *************************/

 var TestPhase = function(){
 	// STATE.set_index(120);
	var that = this;
	// console.log("that",that)
	var clickPositions = [];
	// Initialize a new trial. This is called either at the beginning
	// of a new trial, or if the page is reloaded between trials.
	this.init_trial = function () {
		// If there are no more trials left, then we are at the end of
		// this phase
		if (STATE.index >= $c.trials.length) { //change here for debugging
			this.finish();
			return false;
		}

		// Load the new trialinfo
		this.trialinfo = $c.trials[STATE.index];

		// debugger
		// Update progress bar
		update_progress(STATE.index, $c.trials.length);

		return true;
	}; 


	this.display_stim = function (that) {
		//reset values
		var clickCounter = 0;
		var maxClicks = 10;
		var that = this;
		clickPositions = [];
		// if (that.init_trial()) {
		$('#clicks_remaining').html('You have 10 clicks left.')
		
		if (this.init_trial()) {
			// Question prompt 
			html = "";
			var q = $c.question;
			html += '<p class=".question">' + q + '</div>';
			$('#question_container').html(html);

			//Images 
			var image_path = "static/images/stimuli/" + that.trialinfo.image  + ".jpg"
			$('#stimulus_image').attr("src", image_path);

			// Response measure  
			$(this).ready(function(){
				$('#special').attr('height', $('#special').css('height'));
				$('#special').attr('width', $('#special').css('width'));
				
				//sets offset relative to element
				var offset = $('#image_container').offset()
				
				$("#special").click(function(e){ 
					if (clickCounter <= maxClicks){
					var x = e.pageX - offset.left;
					console.log("x", x);
					var y = e.pageY - offset.top;
					var circleSize = 20;
					var y = parseInt($('#special').attr('height'),10)-(circleSize+2)-20; 
					// var y = parseInt($('#special').attr('height')-20,10)-(circleSize+2); 
					var ctx= this.getContext("2d"); /*c.getContext("2d");*/
					ctx.beginPath();
					ctx.arc(x, y, circleSize,0, 2*Math.PI);
					// ctx.stroke();
					ctx.fillStyle = "red";
					ctx.globalAlpha = 0.3;
					ctx.fill();
					$('#status2').html(x +', '+ y); 
					clickPositions.push(x)
					clickCounter++
					}
					if (clickCounter == maxClicks){
						$("#special").unbind();
						setTimeout(that.record_response,1000);
					}
					var html = 'You have ' + (maxClicks-clickCounter)+ ' clicks left.'
					$('#clicks_remaining').html(html)
				});
			})  
		}       
	};

	this.record_response = function() {      

		var data = {
			trial: that.trialinfo.trial,
			image: that.trialinfo.image,
			clicks: clickPositions.toString()
		}

		// console.log(data)
		psiTurk.recordTrialData(data)

		STATE.set_index(STATE.index + 1);

		// Update the page with the current phase/trial
		// this.display_stim(this);
		that.display_stim(this);
		// this.display_stim(that);
	};

	this.finish = function() {
		// CURRENTVIEW = new Comprehension();
		CURRENTVIEW = new Demographics();
	};

	// Load the trial html page
	$(".slide").hide();

	// Show the slide
	var that = this; 
	$("#trial").fadeIn($c.fade);
	$('#trial_next.next').click(function () {
		that.record_response();
	});

	// Initialize the current trial
	if (this.init_trial()) {
		// Start the test
		this.display_stim(this);
	};
}


/*****************
 *  DEMOGRAPHICS*
 *****************/

 var Demographics = function(){

	var that = this; 

// Show the slide
$(".slide").hide();
$("#demographics").fadeIn($c.fade);

	//disable button initially
	$('#trial_finish').prop('disabled', true);

	//checks whether all questions were answered
	$('.demoQ').change(function () {
	   if ($('input[name=sex]:checked').length > 0 &&
		 $('input[name=age]').val() != "")
	   {
		$('#trial_finish').prop('disabled', false)
	}else{
		$('#trial_finish').prop('disabled', true)
	}
});

// deletes additional values in the number fields 
$('.numberQ').change(function (e) {    
	if($(e.target).val() > 100){
		$(e.target).val(100)
	}
});

this.finish = function() {
	debug("Finish test phase");

		// Show a page saying that the HIT is resubmitting, and
		// show the error page again if it times out or error
		var resubmit = function() {
			$(".slide").hide();
			$("#resubmit_slide").fadeIn($c.fade);

			var reprompt = setTimeout(prompt_resubmit, 10000);
			psiTurk.saveData({
				success: function() {
					clearInterval(reprompt); 
					finish();
				}, 
				error: prompt_resubmit
			});
		};

		// Prompt them to resubmit the HIT, because it failed the first time
		var prompt_resubmit = function() {
			$("#resubmit_slide").click(resubmit);
			$(".slide").hide();
			$("#submit_error_slide").fadeIn($c.fade);
		};

		// Render a page saying it's submitting
		psiTurk.showPage("submit.html");
		psiTurk.saveData({
			success: psiTurk.completeHIT, 
			error: prompt_resubmit
		});
	}; //this.finish function end 

	$('#trial_finish').click(function () {           
	   var feedback = $('textarea[name = feedback]').val();
	   var sex = $('input[name=sex]:checked').val();
	   var age = $('input[name=age]').val();

	   psiTurk.recordUnstructuredData('feedback',feedback);
	   psiTurk.recordUnstructuredData('sex',sex);
	   psiTurk.recordUnstructuredData('age',age);
	   that.finish();
   });
};


// --------------------------------------------------------------------

/*******************
 * Run Task
 ******************/

 $(document).ready(function() { 
	// Load the HTML for the trials
	psiTurk.showPage("trial.html");

	// Record various unstructured data
	psiTurk.recordUnstructuredData("condition", condition);
	psiTurk.recordUnstructuredData("counterbalance", counterbalance);

	// Start the experiment
	STATE = new State();
	// Begin the experiment phase
	if (STATE.instructions) {
		CURRENTVIEW = new Instructions();
	}
});
