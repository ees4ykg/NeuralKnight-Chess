import pygame
import chess
from predict import predict_move

# Initialize Pygame
pygame.init()

# Constants
WIDTH = 800
HEIGHT = 800
SQUARE_SIZE = WIDTH // 8
FPS = 60

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (186, 202, 68)
SELECTED = (246, 246, 105)

# Piece colors
LIGHT_PIECE = (255, 255, 255)
DARK_PIECE = (0, 0, 0)
OUTLINE = (50, 50, 50)


def draw_pawn(surface, x, y, size, is_white):
    """Draw a pawn piece"""
    color = LIGHT_PIECE if is_white else DARK_PIECE
    center_x = x + size // 2
    center_y = y + size // 2
    
    # Circle head
    pygame.draw.circle(surface, color, (center_x, center_y - size // 6), size // 6)
    pygame.draw.circle(surface, OUTLINE, (center_x, center_y - size // 6), size // 6, 2)
    
    # Body
    pygame.draw.ellipse(surface, color, (center_x - size // 8, center_y - size // 12, size // 4, size // 3))
    pygame.draw.ellipse(surface, OUTLINE, (center_x - size // 8, center_y - size // 12, size // 4, size // 3), 2)
    
    # Base
    pygame.draw.rect(surface, color, (center_x - size // 5, center_y + size // 5, size // 2.5, size // 10))
    pygame.draw.rect(surface, OUTLINE, (center_x - size // 5, center_y + size // 5, size // 2.5, size // 10), 2)


def draw_rook(surface, x, y, size, is_white):
    """Draw a rook piece"""
    color = LIGHT_PIECE if is_white else DARK_PIECE
    center_x = x + size // 2
    center_y = y + size // 2
    
    # Top crenellations
    for i in range(3):
        rect_x = center_x - size // 5 + i * size // 7
        pygame.draw.rect(surface, color, (rect_x, center_y - size // 3, size // 8, size // 6))
        pygame.draw.rect(surface, OUTLINE, (rect_x, center_y - size // 3, size // 8, size // 6), 2)
    
    # Body
    pygame.draw.rect(surface, color, (center_x - size // 6, center_y - size // 6, size // 3, size // 2.5))
    pygame.draw.rect(surface, OUTLINE, (center_x - size // 6, center_y - size // 6, size // 3, size // 2.5), 2)
    
    # Base
    pygame.draw.rect(surface, color, (center_x - size // 4, center_y + size // 6, size // 2, size // 10))
    pygame.draw.rect(surface, OUTLINE, (center_x - size // 4, center_y + size // 6, size // 2, size // 10), 2)


def draw_knight(surface, x, y, size, is_white):
    """Draw a knight piece"""
    color = LIGHT_PIECE if is_white else DARK_PIECE
    center_x = x + size // 2
    center_y = y + size // 2
    
    # Horse head (simplified L shape)
    points = [
        (center_x - size // 8, center_y + size // 6),
        (center_x - size // 8, center_y - size // 12),
        (center_x, center_y - size // 3),
        (center_x + size // 6, center_y - size // 6),
        (center_x + size // 6, center_y + size // 6)
    ]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, OUTLINE, points, 2)
    
    # Eye
    pygame.draw.circle(surface, OUTLINE, (center_x + size // 12, center_y - size // 8), 2)
    
    # Base
    pygame.draw.rect(surface, color, (center_x - size // 4, center_y + size // 6, size // 2, size // 10))
    pygame.draw.rect(surface, OUTLINE, (center_x - size // 4, center_y + size // 6, size // 2, size // 10), 2)


def draw_bishop(surface, x, y, size, is_white):
    """Draw a bishop piece"""
    color = LIGHT_PIECE if is_white else DARK_PIECE
    center_x = x + size // 2
    center_y = y + size // 2
    
    # Top circle with cross
    pygame.draw.circle(surface, color, (center_x, center_y - size // 4), size // 12)
    pygame.draw.circle(surface, OUTLINE, (center_x, center_y - size // 4), size // 12, 2)
    pygame.draw.line(surface, OUTLINE, (center_x, center_y - size // 3), (center_x, center_y - size // 6), 2)
    
    # Body (teardrop shape)
    points = [
        (center_x, center_y - size // 4),
        (center_x - size // 7, center_y - size // 8),
        (center_x - size // 8, center_y + size // 8),
        (center_x, center_y + size // 6),
        (center_x + size // 8, center_y + size // 8),
        (center_x + size // 7, center_y - size // 8)
    ]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, OUTLINE, points, 2)
    
    # Base
    pygame.draw.rect(surface, color, (center_x - size // 4, center_y + size // 6, size // 2, size // 10))
    pygame.draw.rect(surface, OUTLINE, (center_x - size // 4, center_y + size // 6, size // 2, size // 10), 2)


def draw_queen(surface, x, y, size, is_white):
    """Draw a queen piece"""
    color = LIGHT_PIECE if is_white else DARK_PIECE
    center_x = x + size // 2
    center_y = y + size // 2
    
    # Crown with multiple points
    crown_points = [
        (center_x, center_y - size // 3),
        (center_x - size // 8, center_y - size // 5),
        (center_x - size // 6, center_y - size // 3.5),
        (center_x - size // 4, center_y - size // 5),
        (center_x - size // 5, center_y - size // 8),
        (center_x, center_y - size // 12),
        (center_x + size // 5, center_y - size // 8),
        (center_x + size // 4, center_y - size // 5),
        (center_x + size // 6, center_y - size // 3.5),
        (center_x + size // 8, center_y - size // 5)
    ]
    pygame.draw.polygon(surface, color, crown_points)
    pygame.draw.polygon(surface, OUTLINE, crown_points, 2)
    
    # Body
    points = [
        (center_x - size // 6, center_y - size // 12),
        (center_x - size // 8, center_y + size // 8),
        (center_x + size // 8, center_y + size // 8),
        (center_x + size // 6, center_y - size // 12)
    ]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, OUTLINE, points, 2)
    
    # Base
    pygame.draw.rect(surface, color, (center_x - size // 4, center_y + size // 8, size // 2, size // 10))
    pygame.draw.rect(surface, OUTLINE, (center_x - size // 4, center_y + size // 8, size // 2, size // 10), 2)


def draw_king(surface, x, y, size, is_white):
    """Draw a king piece"""
    color = LIGHT_PIECE if is_white else DARK_PIECE
    center_x = x + size // 2
    center_y = y + size // 2
    
    # Cross on top
    cross_size = size // 8
    pygame.draw.line(surface, color, (center_x, center_y - size // 2.5), (center_x, center_y - size // 4), 4)
    pygame.draw.line(surface, OUTLINE, (center_x, center_y - size // 2.5), (center_x, center_y - size // 4), 2)
    pygame.draw.line(surface, color, (center_x - cross_size // 2, center_y - size // 3), 
                     (center_x + cross_size // 2, center_y - size // 3), 4)
    pygame.draw.line(surface, OUTLINE, (center_x - cross_size // 2, center_y - size // 3), 
                     (center_x + cross_size // 2, center_y - size // 3), 2)
    
    # Crown
    pygame.draw.rect(surface, color, (center_x - size // 5, center_y - size // 4, size // 2.5, size // 8))
    pygame.draw.rect(surface, OUTLINE, (center_x - size // 5, center_y - size // 4, size // 2.5, size // 8), 2)
    
    # Body
    points = [
        (center_x - size // 5, center_y - size // 6),
        (center_x - size // 7, center_y + size // 8),
        (center_x + size // 7, center_y + size // 8),
        (center_x + size // 5, center_y - size // 6)
    ]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, OUTLINE, points, 2)
    
    # Base
    pygame.draw.rect(surface, color, (center_x - size // 4, center_y + size // 8, size // 2, size // 10))
    pygame.draw.rect(surface, OUTLINE, (center_x - size // 4, center_y + size // 8, size // 2, size // 10), 2)


# Piece drawing function map
PIECE_DRAWERS = {
    'P': draw_pawn,
    'R': draw_rook,
    'N': draw_knight,
    'B': draw_bishop,
    'Q': draw_queen,
    'K': draw_king,
    'p': draw_pawn,
    'r': draw_rook,
    'n': draw_knight,
    'b': draw_bishop,
    'q': draw_queen,
    'k': draw_king
}


class ChessGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Neural Knight Chess - Play vs AI")
        self.clock = pygame.time.Clock()
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.small_font = pygame.font.Font(None, 36)
        self.game_over = False
        self.message = ""
        
    def square_to_coords(self, square):
        """Convert chess square index to pixel coordinates"""
        file = chess.square_file(square)
        rank = 7 - chess.square_rank(square)  # Flip rank for display
        return file * SQUARE_SIZE, rank * SQUARE_SIZE
    
    def coords_to_square(self, x, y):
        """Convert pixel coordinates to chess square index"""
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)  # Flip rank for display
        return chess.square(file, rank)
    
    def draw_board(self):
        """Draw the chess board"""
        for rank in range(8):
            for file in range(8):
                # Determine square color
                is_light = (rank + file) % 2 == 0
                color = WHITE if is_light else BLACK
                
                # Draw square
                rect = pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, 
                                   SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
    
    def draw_highlights(self):
        """Draw highlights for selected square and legal moves"""
        if self.selected_square is not None:
            x, y = self.square_to_coords(self.selected_square)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            s.set_alpha(128)
            s.fill(SELECTED)
            self.screen.blit(s, (x, y))
            
            # Highlight legal move destinations
            for move in self.legal_moves:
                if move.from_square == self.selected_square:
                    x, y = self.square_to_coords(move.to_square)
                    s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                    s.set_alpha(128)
                    s.fill(HIGHLIGHT)
                    self.screen.blit(s, (x, y))
    
    def draw_pieces(self):
        """Draw all pieces on the board"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x, y = self.square_to_coords(square)
                is_white = piece.color == chess.WHITE
                draw_func = PIECE_DRAWERS[piece.symbol()]
                draw_func(self.screen, x, y, SQUARE_SIZE, is_white)
    
    def draw_info(self):
        """Draw game information"""
        turn_text = "White's turn" if self.board.turn == chess.WHITE else "Black's turn (AI thinking...)"
        if self.game_over:
            turn_text = self.message
        
        # Draw semi-transparent background for text
        info_surface = pygame.Surface((WIDTH, 40))
        info_surface.set_alpha(200)
        info_surface.fill((50, 50, 50))
        self.screen.blit(info_surface, (0, 0))
        
        # Draw text
        text = self.small_font.render(turn_text, True, (255, 255, 255))
        self.screen.blit(text, (10, 8))
    
    def handle_click(self, pos):
        """Handle mouse click on the board"""
        if self.game_over:
            return
        
        # Only allow user to move on white's turn
        if self.board.turn != chess.WHITE:
            return
        
        x, y = pos
        clicked_square = self.coords_to_square(x, y)
        
        # If a square is already selected, try to move
        if self.selected_square is not None:
            move = None
            # Check if the clicked square is a legal move destination
            for legal_move in self.legal_moves:
                if (legal_move.from_square == self.selected_square and 
                    legal_move.to_square == clicked_square):
                    move = legal_move
                    break
            
            if move:
                # Handle pawn promotion
                if move.promotion is None and self.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                    # Check if it's a promotion move
                    if (chess.square_rank(clicked_square) == 7 or 
                        chess.square_rank(clicked_square) == 0):
                        # Default to queen promotion
                        move = chess.Move(self.selected_square, clicked_square, promotion=chess.QUEEN)
                
                # Make the move
                self.board.push(move)
                self.selected_square = None
                self.legal_moves = []
                self.check_game_over()
                
                # If game not over, let AI make a move
                if not self.game_over:
                    pygame.display.flip()
                    self.ai_move()
                
                return
            
            # If clicked on a different piece of the same color, select it
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == chess.WHITE:
                self.selected_square = clicked_square
                self.legal_moves = [move for move in self.board.legal_moves 
                                   if move.from_square == clicked_square]
            else:
                # Deselect
                self.selected_square = None
                self.legal_moves = []
        else:
            # Select a piece if it's white
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == chess.WHITE:
                self.selected_square = clicked_square
                self.legal_moves = [move for move in self.board.legal_moves 
                                   if move.from_square == clicked_square]
    
    def ai_move(self):
        """Let the AI make a move"""
        if self.board.turn == chess.BLACK and not self.game_over:
            # Redraw to show "AI thinking" message
            self.draw()
            pygame.display.flip()
            
            move_uci = predict_move(self.board)
            if move_uci:
                move = chess.Move.from_uci(move_uci)
                self.board.push(move)
                self.check_game_over()
            else:
                self.game_over = True
                self.message = "AI couldn't find a move!"
    
    def check_game_over(self):
        """Check if the game is over and set appropriate message"""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.message = f"Checkmate! {winner} wins!"
            self.game_over = True
        elif self.board.is_stalemate():
            self.message = "Stalemate! Draw!"
            self.game_over = True
        elif self.board.is_insufficient_material():
            self.message = "Draw by insufficient material!"
            self.game_over = True
        elif self.board.is_fifty_moves():
            self.message = "Draw by fifty-move rule!"
            self.game_over = True
        elif self.board.is_repetition():
            self.message = "Draw by repetition!"
            self.game_over = True
        elif self.board.is_check():
            # Just a check, not game over
            pass
    
    def draw(self):
        """Draw everything"""
        self.draw_board()
        self.draw_highlights()
        self.draw_pieces()
        self.draw_info()
    
    def run(self):
        """Main game loop"""
        running = True
        
        print("Neural Knight Chess GUI")
        print("You are playing as White")
        print("Click on a piece to select it, then click on a square to move")
        print("Close the window to exit")
        
        while running:
            self.clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
            
            self.draw()
            pygame.display.flip()
        
        pygame.quit()


if __name__ == "__main__":
    game = ChessGUI()
    game.run()

