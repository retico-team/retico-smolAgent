import threading
import logging
import re
from typing import List

from smolagents import CodeAgent
import retico_core
from retico_core import abstract
from retico_core.text import SpeechRecognitionIU, TextIU


class SmolAgentsModule(abstract.AbstractModule):
    """
    SmolAgents module for Retico framework
    Acts as NLU + Dialog Manager + NLG using CodeAgent
    """
    
    def __init__(self, agent: CodeAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.processing = False
        self.lock = threading.Lock()
        self.logger = logging.getLogger(self.name())
        self.conversation_context = []
    
    @staticmethod
    def name() -> str:
        return "SmolAgents"
    
    @staticmethod
    def description() -> str:
        return "SmolAgents module for conversational using CodeAgent"
    
    @staticmethod
    def input_ius() -> List[type]:  
        return [SpeechRecognitionIU]
    
    @staticmethod
    def output_iu():
        return TextIU
    
    def process_update(self, um) -> None:
        """Process incoming speech recognition updates"""
        if self.processing:
            return
            
        # Extract text from all committed speech recognition IUs
        user_text_parts = []
        source_iu = None
        
        for iu, ut in um:
            if (ut == abstract.UpdateType.COMMIT and 
                isinstance(iu, SpeechRecognitionIU)):
                if iu.text and iu.text.strip():
                    user_text_parts.append(iu.text.strip())
                source_iu = iu  # Keep the last one as source
        
        # Process if we have text and source
        if user_text_parts and source_iu:
            complete_text = " ".join(user_text_parts)
            self.logger.info(f"Processing: '{complete_text}'")
            
            # Start processing in separate thread
            threading.Thread(
                target=self._process_with_agent, 
                args=(source_iu, complete_text), 
                daemon=True
            ).start()
    

    def _process_with_agent(self, source_iu: SpeechRecognitionIU, user_input: str) -> None:
        """Process user input with SmolAgent and send response"""
        with self.lock:
            
            self.processing = True
            
            # Build context prompt
            if self.conversation_context:
                recent_context = "\n".join(self.conversation_context[-8:])  # Last 8 exchanges
                context_prompt = f"""Recent conversation:{recent_context}
                                Current user input: {user_input}"""
            else:
                context_prompt = user_input
            
                # Get agent response
                
                raw_response = self.agent.run(context_prompt)
                clean_response = self._clean_response(raw_response)
                self.logger.info(f"Response: '{clean_response}'")
                
                # Update conversation history
                self.conversation_context.extend([
                    f"User: {user_input}",
                    f"System: {clean_response}"
                ])
                
            # Keep only last 16 exchanges (8 user + 8 assistant)
            if len(self.conversation_context) > 16:
                self.conversation_context = self.conversation_context[-16:]
            
                
            # Send response word by word
            self._send_response(source_iu, clean_response)  
            self.processing = False
    
    
    def _clean_response(self, response) -> str:
        """Clean agent response for speech output"""
        if not response:
            return "I didn't understand that."
        
        text = str(response)
        
        # Handle dictionary responses
        if isinstance(response, dict):
            items = [f"{k}: {v}" for k, v in response.items()]
            return f"I have collected your information: {', '.join(items)}"
        
        # Extract from final_answer() or print() calls
        patterns = [
            r'final_answer\s*\(["\']([^"\']*)["\']',
            r'print\s*\(["\']([^"\']*)["\']'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Clean raw text (keep essential punctuation)
        clean = text.strip()
        return clean if clean else "I'm processing that."
    

    
    def _send_response(self, source_iu, text: str):
        """Send response as incremental TextIUs word by word"""
        
        words = text.split()
        if not words:
            return
        
        output_ius = []
        
        # Send each word as ADD message
        for word in words:
            output_iu = self.create_iu(source_iu)
            output_iu.text = word
            output_iu.payload =word
            output_ius.append(output_iu)
            
            
            add_msg = retico_core.UpdateMessage.from_iu(
                output_iu, 
                retico_core.UpdateType.ADD
            )
            self.append(add_msg)

        
        # Send final COMMIT message
        if output_ius:
                last_iu = output_ius[-1]
                commit_msg = retico_core.UpdateMessage.from_iu(
                    last_iu,
                    retico_core.UpdateType.COMMIT
                )
                self.append(commit_msg)
            
        
    
    def setup(self) -> None:
        super().setup()
        self.logger.info("SmolAgents module ready")
    

    def shutdown(self) -> None:
        super().shutdown()
        self.processing = False
        self.logger.info("SmolAgents module shutdown")