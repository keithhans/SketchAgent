import utils
import math
import ast
import cairosvg
import os
from dotenv import load_dotenv
import anthropic
from prompts import sketch_first_prompt, system_prompt, gt_example
import json
import socket
from flask import Flask, render_template, request, jsonify
import time
import signal
import random
from datetime import datetime
import traceback
import uuid
from PIL import Image


class SketchApp:
    """
    A Python class that manages the interactive drawing process.
    This class should be used when a sketching session is initialized. Here, we keep track on the sketching history, and call our sketching agent to draw sequential strokes with the user.
    """
    def __init__(self, res, cell_size, grid_size, stroke_width, target_concept, user_always_first):
        self.app = Flask(__name__)
        self.session_id = str(uuid.uuid4())

        # LLM Setup (you need to provide your ANTHROPIC_API_KEY in your .env file)
        self.seed_mode = "stochastic"
        self.cache = False
        self.max_tokens = 3000
        load_dotenv()
        claude_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=claude_key)
        self.model = "claude-3-5-sonnet-20240620"

        # Grid setup
        self.res = res
        self.num_cells = res
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.init_canvas_grid, self.positions = utils.create_grid_image(res=res, cell_size=cell_size, header_size=cell_size)
        self.init_canvas = Image.new('RGB', self.grid_size, 'white')
        self.init_canvas.save("static/init_canvas.png")
        self.stroke_width = stroke_width
        self.num_sampled_points = 100

        # Program init
        self.user_always_first = user_always_first
        self.folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.drawn_concepts = []
        
        self.target_concept = target_concept
        self.sketch_mode = "solo"
        self.cur_svg_to_render = "None"
        self.initialize_all()

        # Define routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/update-mode', 'set_sketch_mode', self.set_sketch_mode, methods=['POST'])
        self.app.add_url_rule('/send-user-strokes', 'get_user_stroke', self.get_user_stroke, methods=['POST'])
        self.app.add_url_rule('/call-agent', 'call_agent', self.call_agent, methods=['POST'])
        self.app.add_url_rule('/clear-canvas', 'clear_canvas', self.clear_canvas, methods=['POST'])
        self.app.add_url_rule('/submit-sketch', 'submit_sketch', self.submit_sketch, methods=['POST'])
        self.app.add_url_rule('/get-new-concept', 'get_new_concept', self.get_new_concept, methods=['POST'])
        self.app.add_url_rule('/draw-sketch', 'draw_sketch', self.draw_entire_sketch, methods=['POST'])
        self.app.add_url_rule('/shutdown', 'shutdown', self.shutdown, methods=['POST'])
    
    def get_agent_svg(self):
        print("get_agent_svg==============")
        return self.cur_svg_to_render
        
        # return self.cur_svg_to_render

    def set_sketch_mode(self):
        data = request.get_json()
        new_sketch_mode = data.get("mode", "solo")
        self.sketch_mode = new_sketch_mode
        self.init_canvas.save("static/init_canvas.png")
        return jsonify({"status": "success", "message": f"Mode set to {self.sketch_mode}"})


    def setup_path2save(self):
        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path2save = f"results/collab_sketching/{self.folder_name}_{self.session_id}/{self.target_concept}/{self.sketch_mode}_{folder_name}"
        if not os.path.exists(self.path2save):
            os.makedirs(self.path2save)
        with open(f"{self.path2save}/data_history.json", "w") as f:
            json.dump([{"session_ID": self.session_id}], f)


    def initialize_all(self):
        self.input_prompt = sketch_first_prompt.format(concept=self.target_concept, gt_sketches_str=gt_example)
        self.all_strokes_svg = f"""<svg width="{self.grid_size[0]}" height="{self.grid_size[1]}" xmlns="http://www.w3.org/2000/svg">"""
        self.assitant_history = ""
        self.stroke_counter = 0
        self.setup_path2save()
        self.init_canvas.save("static/cur_canvas_user.png")
        self.init_canvas.save("static/cur_canvas_agent.png")
        self.init_canvas.save("static/init_canvas.png")
        if self.sketch_mode == "colab":
            self.init_thinking_tags()
            print("Thinking Ready!")


    def get_new_concept(self):
        """
        Sample a new concept to sketch, this should re-initialize the entire system!
        """
        data = request.get_json()
        self.target_concept = data.get('concept')  # Get the user name
        self.initialize_all()
        return jsonify({"target_concept": self.target_concept, "SVG": self.get_agent_svg()})
    

    def submit_sketch(self):
        self.all_strokes_svg += "</svg>"
        with open(f"{self.path2save}/final_sketch.svg", "w") as svg_file:
            svg_file.write(self.all_strokes_svg)
        cairosvg.svg2png(url=f"{self.path2save}/final_sketch.svg", write_to=f"{self.path2save}/final_sketch.png", background_color="white")
        print(f"results saved to [{self.path2save}/final_sketch.svg]")
        # Load the existing JSON data
        with open(f"{self.path2save}/data_history.json", "r") as f:
            data = json.load(f)
            data.append({f"all_history": self.assitant_history})
        with open(f"{self.path2save}/data_history.json", "w") as f:
            json.dump(data, f)
        return jsonify({"new_category": "yes", "mode": "colab", "message": f"Sketch saved! Continue to next concept!"})
       

    def clear_canvas(self, same_session=True):
        self.initialize_all()
        # delete current sketch from "static"
        self.init_canvas.save("static/cur_canvas_user.png")
        self.init_canvas.save("static/cur_canvas_agent.png")
        if same_session:
            print(f"removing {self.path2save}/sketch.svg")
            if os.path.exists(f"{self.path2save}/sketch.svg"):
                print(f"removing {self.path2save}/sketch.svg")
                os.remove(f"{self.path2save}/sketch.svg")
        return jsonify({"message": f"cleaned!"}) 


    def index(self):
        return render_template('index.html', target_concept=self.target_concept)

    def shutdown(self):
        self.shutdown_server()
        return 'Server shutting down...'

    def shutdown_server(self):
        # This function sends a kill signal to shut down the Flask app
        os.kill(os.getpid(), signal.SIGINT)

    def update_history(self, txt_update, replace=False):
        if replace:
            self.assitant_history = txt_update
        else:
            self.assitant_history += txt_update

        # Load the existing JSON data
        with open(f"{self.path2save}/data_history.json", "r") as f:
            data = json.load(f)
            data.append({f"stroke_{self.stroke_counter}": self.assitant_history})
        
        with open(f"{self.path2save}/data_history.json", "w") as f:
            json.dump(data, f)
    
    def get_user_stroke(self):
        # Receive the strokes data from the frontend
        try:
            data = request.get_json()
            self.user_name = data.get('name')  # Get the user name
            sketch_data = data.get('strokes')  # Get the strokes data
            # make sure data recieved as expected from user:
            assert len(sketch_data[0]) > 0, "No strokes provided."
            
            self.stroke_counter += 1
            try:
                user_stroke = self.parse_stroke_from_canvas(sketch_data) # saves the stroke's string in self.str_rep_strokes
                user_stroke_svg = self.parse_model_to_svg(f"{user_stroke}</s{self.stroke_counter}>")
            except Exception as e:
                print(sketch_data)
                print(f"An error has occurred: {e}")
                traceback.print_exc()
                self.stroke_counter -= 1
                return jsonify({"message": str(e), "status": "error"}), 400
            
            self.all_strokes_svg += user_stroke_svg
            cur_svg_to_render = f"{self.all_strokes_svg}</svg>"
            with open(f"{self.path2save}/sketch.svg", "w") as svg_file:
                svg_file.write(cur_svg_to_render)

            # 2. Convert the SVG file to PNG (or another image format) using CairoSVG
            cairosvg.svg2png(url=f"{self.path2save}/sketch.svg", write_to=f"static/cur_canvas_user.png", background_color="white")
            
            self.update_history(user_stroke)
            if self.sketch_mode == "solo":
                self.update_history(f"</s{self.stroke_counter}>")
            return jsonify({"message": "User strokes received successfully!"})
        
        except Exception as e:
            print(sketch_data)
            print(f"An error has occurred: {e}")
            traceback.print_exc()
            return jsonify({"message": str(e), "status": "error"}), 400

        
    
    def call_agent(self):
        print("Calling LLM...!")
        try:
            model_stroke_svg = self.predict_next_stroke()
            self.all_strokes_svg += model_stroke_svg
            self.cur_svg_to_render = f"{self.all_strokes_svg}</svg>"
            with open(f"{self.path2save}/sketch.svg", "w") as svg_file:
                svg_file.write(self.cur_svg_to_render)
            cairosvg.svg2png(url=f"{self.path2save}/sketch.svg", write_to=f"static/cur_canvas_agent.png", background_color="white")
            if not self.user_always_first:
                cairosvg.svg2png(url=f"{self.path2save}/sketch.svg", write_to=f"static/init_canvas.png", background_color="white")
            return jsonify({"status": "success", "SVG": self.cur_svg_to_render})
        
        except Exception as e:
            print(f"An error has occurred: {e}")
            traceback.print_exc()
            return jsonify({"message": str(e), "status": "error"}), 400
        


    def parse_stroke_from_canvas(self, sketch_data):
        cur_user_input_stroke = f"<s{self.stroke_counter}>\n" # for first user input
        stroke = sketch_data[0] # assume one stroke from user at a time
        cur_user_input_stroke += f"<points>"

        cur_stroke = []
        cur_t_values = []
        for point_data in stroke:
            x, y, t = point_data['x'], point_data['y'], point_data['timestamp']
            x = min(self.grid_size[0] - 1, max(self.cell_size, x))  # Constrain x between 0 and 599
            y = min(self.grid_size[0] - 1 - self.cell_size, max(0, y))  # Constrain y between 0 and 599
            
            # Change to textual representation
            grid_x = int(x // self.cell_size) #+ 1
            grid_y = int(self.num_cells - (y // self.cell_size))
            point_str = f'x{grid_x}y{grid_y}'
    
            # Calculate the distance from the current point to the center of the grid cell
            cell_center = self.positions[point_str]
            distance = math.sqrt((x - cell_center[0]) ** 2 + (y - cell_center[1]) ** 2)
            
            # print("distance", distance)
            if distance <= 5:
                # Check if the point is new, and add it to the current stroke list
                if (not cur_stroke) or (cur_stroke[-1] != point_str):
                    cur_stroke.append(point_str)
                    cur_t_values.append(t)
                    cur_user_input_stroke += f"'{point_str}', "
        
        if len(cur_t_values) == 0:
            for point_data in stroke:
                x, y, t = point_data['x'], point_data['y'], point_data['timestamp']
                x = min(self.grid_size[0] - 1, max(self.cell_size, x))  # Constrain x between 0 and 599
                y = min(self.grid_size[0] - 1 - self.cell_size, max(0, y))  # Constrain y between 0 and 599
                
                # Change to textual representation
                grid_x = int(x // self.cell_size) #+ 1
                grid_y = int(self.num_cells - (y // self.cell_size))
                point_str = f'x{grid_x}y{grid_y}'
        
                # Calculate the distance from the current point to the center of the grid cell
                cell_center = self.positions[point_str]
                distance = math.sqrt((x - cell_center[0]) ** 2 + (y - cell_center[1]) ** 2)
                
                # print("distance", distance)
                if distance <= 8:
                    # Check if the point is new, and add it to the current stroke list
                    if (not cur_stroke) or (cur_stroke[-1] != point_str):
                        cur_stroke.append(point_str)
                        cur_t_values.append(t)
                        cur_user_input_stroke += f"'{point_str}', "
        assert len(cur_t_values) > 0, "No values recorded from strokes!"
        cur_user_input_stroke = cur_user_input_stroke[:-2]
        cur_user_input_stroke += "</points>\n"
        cur_user_input_stroke += "<t_values>"
        normalized_ts = []
        min_time = min(cur_t_values)
        max_time = max(cur_t_values)
        for t in cur_t_values:
            cur_n_t = (t - min_time) / (max_time - min_time) if max_time > min_time else 0.0
            normalized_ts.append(float(f"{cur_n_t:.2f}"))
            cur_user_input_stroke += f"{cur_n_t:.2f}, "
        cur_user_input_stroke = cur_user_input_stroke[:-2]
        cur_user_input_stroke += "</t_values>"
        return cur_user_input_stroke


    def call_llm(self, system_message, other_msg, additional_args):
        if self.cache:
            init_response = self.client.beta.prompt_caching.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_message,
                    messages=other_msg,
                    **additional_args
                )
        else:
            init_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_message,
                    messages=other_msg,
                    **additional_args
                )
        return init_response

    
    def define_input_to_llm(self, msg_history, init_canvas_str, msg):
        # other_msg should contain all messgae without the system prompt
        other_msg = msg_history 

        content = []
        # Claude best practice is image-then-text
        if init_canvas_str is not None:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": init_canvas_str}}) 

        content.append({"type": "text", "text": msg})
        if self.cache:
            content[-1]["cache_control"] = {"type": "ephemeral"}

        other_msg = other_msg + [{"role": "user", "content": content}]
        return other_msg


    def get_response_from_llm(
        self,
        msg,
        system_message,
        msg_history=[],
        init_canvas_str=None,
        prefill_msg=None,
        seed_mode="stochastic",
        stop_sequences=None,
        gen_mode="generation"
    ):  
        additional_args = {}
        if seed_mode == "deterministic":
            additional_args["temperature"] = 0.0
            additional_args["top_k"] = 1

        if self.cache:
            system_message = [{
                "type": "text",
                "text": system_message,
                "cache_control": {"type": "ephemeral"}
            }]

        # other_msg should contain all messgae without the system prompt
        other_msg = self.define_input_to_llm(msg_history, init_canvas_str, msg) 

        if gen_mode == "completion":
            if prefill_msg:
                other_msg = other_msg + [{"role": "assistant", "content": f"{prefill_msg}"}]
            
            # in case of stroke by stroke generation
        if stop_sequences:
            additional_args["stop_sequences"]= [stop_sequences]
        else:
            additional_args["stop_sequences"]= ["</answer>"]

        # Note that we deterministic settings for reproducibility (temperature=0.0 and top_k=1). 
        # To run in stochastic mode just comment these parameters.
        response = self.call_llm(system_message, other_msg, additional_args)

        content = response.content[0].text
        
        if gen_mode == "completion":
            other_msg = other_msg[:-1] # remove initial assistant prompt
            content = f"{prefill_msg}{content}" 

        # saves to json
        if self.path2save is not None:
            system_message_json = [{"role": "system", "content": system_message}]
            new_msg_history = other_msg + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }
            ]    
            with open(f"{self.path2save}/experiment_log.json", 'w') as json_file:
                json.dump(system_message_json + new_msg_history, json_file, indent=4)
            print(f"Data has been saved to [{self.path2save}/experiment_log.json]")

        return content

    
    def init_thinking_tags(self):
        print("Init thinking tags...")
        gen_mode = "generation"
        seed_mode = self.seed_mode  # choices=['deterministic', 'stochastic']
        sketcher_msg_history = []      
        add_args = {}

        # first generate the thinking tags for both agents
        add_args["stop_sequences"] = f"<strokes>" 

        msg_history = []
        init_canvas_str = None
        # init_canvas_str = self.init_canvas_str # in case we don't want to insert the empty canvas to the model
        
        assistant_suffix = self.get_response_from_llm(
            msg=self.input_prompt,
            system_message=system_prompt.format(res=self.res),
            msg_history=msg_history,
            init_canvas_str=init_canvas_str,
            seed_mode=seed_mode,
            gen_mode=gen_mode,
            **add_args
        )

        self.thinking_tags = assistant_suffix
        self.thinking_tags += "<strokes>"
        self.update_history(self.thinking_tags)
        print("Done!")
        if not self.user_always_first:
            self.call_agent()
        
    
    def draw_entire_sketch(self):
        gen_mode = "generation"
        seed_mode = self.seed_mode  # choices=['deterministic', 'stochastic']
        sketcher_msg_history = []      
        add_args = {}

        # first generate the thinking tags for both agents
        add_args["stop_sequences"] = f"</answer>" 
        msg_history = []
        init_canvas_str = None # in case we don't want to insert the empty canvas to the model

        print("Call LLM")
        all_sketch = self.get_response_from_llm(
            msg=self.input_prompt,
            system_message=system_prompt.format(res=self.res),
            msg_history=msg_history,
            init_canvas_str=init_canvas_str,
            seed_mode=seed_mode,
            gen_mode=gen_mode,
            **add_args
        )

        # Parse model_rep with xml
        strokes_list_str, t_values_str = utils.parse_xml_string(all_sketch, res=self.res)
        strokes_list, t_values = ast.literal_eval(strokes_list_str), ast.literal_eval(t_values_str)
        
        # extract control points from sampled lists
        all_control_points = utils.get_control_points(strokes_list, t_values, self.positions)

        # define SVG based on control point
        sketch_text_svg = utils.format_svg(all_control_points, dim=self.grid_size, stroke_width=self.stroke_width)
        
        with open(f"{self.path2save}/sketch.svg", "w") as svg_file:
            svg_file.write(sketch_text_svg)

        # 2. Convert the SVG file to PNG (or another image format) using CairoSVG
        cairosvg.svg2png(url=f"{self.path2save}/sketch.svg", write_to=f"static/entire_sketch.png", background_color="white")

        return jsonify({"status": "success", "message": "Sketch drawn!"})


    def restart_cur_group(self):
        self.all_sampled_points = []
        self.sampled_points_grid_txt = []
        self.t_values_grid = []


    def get_cell_center(self, x, y):
        grid_x = int(x // self.cell_size) #+ 1
        grid_y = int(self.num_cells - (y // self.cell_size))
        point_str = f'x{grid_x}y{grid_y}'
        cell_center = self.positions[point_str]
        return cell_center, point_str

    
    def parse_model_to_svg(self, stroke_model):
        # Parse model_rep with xml
        strokes_list_str, t_values_str = utils.parse_xml_string_single_stroke(stroke_model, res=self.res, stroke_counter=self.stroke_counter)
        strokes_list, t_values = ast.literal_eval(strokes_list_str), ast.literal_eval(t_values_str)
        
        # extract control points from sampled lists
        all_control_points = utils.get_control_points_single_stroke(strokes_list, t_values, self.positions)

        # define SVG based on control point
        stroke_color = "green"
        if self.sketch_mode == "colab":
            if self.user_always_first:
                if self.stroke_counter % 2 == 0:
                    stroke_color = "pink"
            else:
                if self.stroke_counter % 2 == 1:
                    stroke_color = "pink"
        sketch_text_svg = utils.format_svg_single_stroke(all_control_points, dim=self.grid_size, stroke_width=self.stroke_width, stroke_counter=self.stroke_counter,stroke_color=stroke_color)
        return sketch_text_svg

    def verify_llm_ouput(self, llm_output):
        if "</strokes>" in llm_output or "</answer>" in llm_output:
            self.update_history(llm_output, replace=True)
            raise Exception("Agent decided that the sketch is finished!") 
        
        

    def call_model_stroke_completion(self):
        print("Calling LLM...")
        gen_mode = "completion"
        seed_mode = self.seed_mode  # choices=['deterministic', 'stochastic']
        sketcher_msg_history = []      
        
        add_args = {}
        add_args["stop_sequences"] = f"</s{self.stroke_counter}>" 

        
        msg_history = []
        init_canvas_str = None # in case we don't want to insert the empty canvas to the model

        all_llm_output = self.get_response_from_llm(
            msg=self.input_prompt,
            system_message=system_prompt.format(res=res),
            msg_history=msg_history,
            init_canvas_str=None,
            seed_mode=seed_mode,
            gen_mode=gen_mode,
            prefill_msg=self.assitant_history.strip(),
            **add_args
        )
        self.verify_llm_ouput(all_llm_output) # this will raise an error

        all_llm_output += f"</s{self.stroke_counter}>"
        self.update_history(all_llm_output, replace=True)
        cur_stroke = utils.get_cur_stroke_text(self.stroke_counter, all_llm_output)
        return cur_stroke


    def predict_next_stroke(self):
        """
        Parameters
        ----------
        user_stroke_svg : string
            The last stroke drawn on the canvas by the user, represented in relative SVG code.

        Returns
        -------
        model predicted stroke in SVG code.
        """
        try:
            self.stroke_counter += 1
            stroke_pred = self.call_model_stroke_completion() # one stroke
            model_stroke_svg = self.parse_model_to_svg(stroke_pred)
            return model_stroke_svg
            
        except Exception as e:
            self.stroke_counter -= 1
            raise Exception(e)
        # take care of agent decided to finish
        

    def run(self, hostname, ip_address):
        # Create a socket to find an available port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', 0))  # Port 0 tells the OS to find an available port
        port = sock.getsockname()[1]  # Get the port number that was assigned
        sock.close()
        
        print(f'Server running at: http://{ip_address}:{port}')
        
        # Run the app with the found port
        self.app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)

# Initialize and run the SketchApp
if __name__ == '__main__':
    # Get the IP address of the machine
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    # Print the access link
    print(f'Server running at: http://{ip_address}:5000')

    user_always_first = False

    res = 50
    cell_size = 12
    grid_size = (612,612)
    stroke_width = cell_size * 0.6

    sketch_app = SketchApp(res=res, 
                            cell_size=cell_size,
                            grid_size=grid_size,
                            stroke_width=stroke_width,
                            target_concept="sailboat",
                            user_always_first=user_always_first)
    
    sketch_app.run(hostname, ip_address)